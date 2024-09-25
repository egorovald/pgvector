#include "postgres.h"

#include <float.h>
#include <math.h>

#include "bitvec.h"
#include "halfutils.h"
#include "halfvec.h"
#include "ivfflat.h"
#include "miscadmin.h"
#include "utils/builtins.h"
#include "utils/datum.h"
#include "utils/memutils.h"
#include "vector.h"
#include "greedy.h"

#include "limits.h"
#include "unistd.h"

/*
 * Initialize with kmeans++
 *
 * https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
 */
static void
InitCenters(Relation index, VectorArray samples, VectorArray centers, float *lowerBound)
{
	FmgrInfo   *procinfo;
	Oid			collation;
	int64		j;
	float	   *weight = palloc(samples->length * sizeof(float));
	int			numCenters = centers->maxlen;
	int			numSamples = samples->length;

	procinfo = index_getprocinfo(index, 1, IVFFLAT_KMEANS_DISTANCE_PROC);
	collation = index->rd_indcollation[0];

	/* Choose an initial center uniformly at random */
	VectorArraySet(centers, 0, VectorArrayGet(samples, RandomInt() % samples->length));
	centers->length++;

	for (j = 0; j < numSamples; j++)
		weight[j] = FLT_MAX;

	for (int i = 0; i < numCenters; i++)
	{
		double		sum;
		double		choice;

		CHECK_FOR_INTERRUPTS();

		sum = 0.0;

		for (j = 0; j < numSamples; j++)
		{
			Datum		vec = PointerGetDatum(VectorArrayGet(samples, j));
			double		distance;

			/* Only need to compute distance for new center */
			/* TODO Use triangle inequality to reduce distance calculations */
			distance = DatumGetFloat8(FunctionCall2Coll(procinfo, collation, vec, PointerGetDatum(VectorArrayGet(centers, i))));

			/* Set lower bound */
			lowerBound[j * numCenters + i] = distance;

			/* Use distance squared for weighted probability distribution */
			distance *= distance;

			if (distance < weight[j])
				weight[j] = distance;

			sum += weight[j];
		}

		/* Only compute lower bound on last iteration */
		if (i + 1 == numCenters)
			break;

		/* Choose new center using weighted probability distribution. */
		choice = sum * RandomDouble();
		for (j = 0; j < numSamples - 1; j++)
		{
			choice -= weight[j];
			if (choice <= 0)
				break;
		}

		VectorArraySet(centers, i + 1, VectorArrayGet(samples, j));
		centers->length++;
	}

	pfree(weight);
}

/*
 * Norm centers
 */
static void
NormCenters(const IvfflatTypeInfo * typeInfo, Oid collation, VectorArray centers)
{
	MemoryContext normCtx = AllocSetContextCreate(CurrentMemoryContext,
												  "Ivfflat norm temporary context",
												  ALLOCSET_DEFAULT_SIZES);
	MemoryContext oldCtx = MemoryContextSwitchTo(normCtx);

	for (int j = 0; j < centers->length; j++)
	{
		Datum		center = PointerGetDatum(VectorArrayGet(centers, j));
		Datum		newCenter = IvfflatNormValue(typeInfo, collation, center);
		Size		size = VARSIZE_ANY(DatumGetPointer(newCenter));

		if (size > centers->itemsize)
			elog(ERROR, "safety check failed");

		memcpy(DatumGetPointer(center), DatumGetPointer(newCenter), size);
		MemoryContextReset(normCtx);
	}

	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(normCtx);
}

/*
 * Quick approach if we have no data
 */
static void
RandomCenters(Relation index, VectorArray centers, const IvfflatTypeInfo * typeInfo)
{
	int			dimensions = centers->dim;
	FmgrInfo   *normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_KMEANS_NORM_PROC);
	Oid			collation = index->rd_indcollation[0];
	float	   *x = (float *) palloc(sizeof(float) * dimensions);

	/* Fill with random data */
	while (centers->length < centers->maxlen)
	{
		Pointer		center = VectorArrayGet(centers, centers->length);

		for (int i = 0; i < dimensions; i++)
			x[i] = (float) RandomDouble();

		typeInfo->updateCenter(center, dimensions, x);

		centers->length++;
	}

	if (normprocinfo != NULL)
		NormCenters(typeInfo, collation, centers);
}

#ifdef IVFFLAT_MEMORY
/*
 * Show memory usage
 */
static void
ShowMemoryUsage(MemoryContext context, Size estimatedSize)
{
#if PG_VERSION_NUM >= 130000
	elog(INFO, "total memory: %zu MB",
		 MemoryContextMemAllocated(context, true) / (1024 * 1024));
#else
	MemoryContextStats(context);
#endif
	elog(INFO, "estimated memory: %zu MB", estimatedSize / (1024 * 1024));
}
#endif

/*
 * Sum centers
 */
static void
SumCenters(VectorArray samples, float *agg, int *closestCenters, const IvfflatTypeInfo * typeInfo)
{
	for (int j = 0; j < samples->length; j++)
	{
		float	   *x = agg + ((int64) closestCenters[j] * samples->dim);

		typeInfo->sumCenter(VectorArrayGet(samples, j), x);
	}
}

/*
 * Update centers
 */
static void
UpdateCenters(float *agg, VectorArray centers, const IvfflatTypeInfo * typeInfo)
{
	for (int j = 0; j < centers->length; j++)
	{
		float	   *x = agg + ((int64) j * centers->dim);

		typeInfo->updateCenter(VectorArrayGet(centers, j), centers->dim, x);
	}
}

/*
 * Compute new centers
 */
static void
ComputeNewCenters(VectorArray samples, float *agg, VectorArray newCenters, int *centerCounts, int *closestCenters, FmgrInfo *normprocinfo, Oid collation, const IvfflatTypeInfo * typeInfo)
{
	int			dimensions = newCenters->dim;
	int			numCenters = newCenters->length;
	int			numSamples = samples->length;

	/* Reset sum and count */
	for (int j = 0; j < numCenters; j++)
	{
		float	   *x = agg + ((int64) j * dimensions);

		for (int k = 0; k < dimensions; k++)
			x[k] = 0.0;

		centerCounts[j] = 0;
	}

	/* Increment sum of closest center */
	SumCenters(samples, agg, closestCenters, typeInfo);

	/* Increment count of closest center */
	for (int j = 0; j < numSamples; j++)
		centerCounts[closestCenters[j]] += 1;

	/* Divide sum by count */
	for (int j = 0; j < numCenters; j++)
	{
		float	   *x = agg + ((int64) j * dimensions);

		if (centerCounts[j] > 0)
		{
			/* Double avoids overflow, but requires more memory */
			/* TODO Update bounds */
			for (int k = 0; k < dimensions; k++)
			{
				if (isinf(x[k]))
					x[k] = x[k] > 0 ? FLT_MAX : -FLT_MAX;
			}

			for (int k = 0; k < dimensions; k++)
				x[k] /= centerCounts[j];
		}
		else
		{
			/* TODO Handle empty centers properly */
			for (int k = 0; k < dimensions; k++)
				x[k] = RandomDouble();
		}
	}

	/* Set new centers */
	UpdateCenters(agg, newCenters, typeInfo);

	/* Normalize if needed */
	if (normprocinfo != NULL)
		NormCenters(typeInfo, collation, newCenters);
}

/*
* LE
* Simple K-means
*/
static void
SimpleKmeans(Relation index, VectorArray samples, VectorArray centers, const IvfflatTypeInfo * typeInfo)
{
	FmgrInfo   *procinfo;
	FmgrInfo   *normprocinfo;
	Oid			collation;
	int			dimensions = centers->dim;
	int			numCenters = centers->maxlen;
	int			numSamples = samples->length;

	int		   *centerCounts;
	int		   *closestCenters;
	VectorArray newCenters;
	float	   *agg;	

	float	   *lowerBound;
	/* Calculate allocation sizes */
	Size		samplesSize = VECTOR_ARRAY_SIZE(samples->maxlen, samples->itemsize);
	Size		centersSize = VECTOR_ARRAY_SIZE(centers->maxlen, centers->itemsize);
	Size		lowerBoundSize = sizeof(float) * numSamples * numCenters;
	Size		centerCountsSize = sizeof(int) * numCenters;
	Size		closestCentersSize = sizeof(int) * numSamples;
	Size		newCentersSize = VECTOR_ARRAY_SIZE(numCenters, centers->itemsize);
	Size		aggSize = sizeof(float) * (int64) numCenters * dimensions;


	/* Calculate total size */
	//Size		totalSize = samplesSize + centersSize + newCentersSize + aggSize + centerCountsSize + closestCentersSize + lowerBoundSize + upperBoundSize + sSize + halfcdistSize + newcdistSize;
	Size		totalSize = samplesSize + centersSize + newCentersSize + aggSize                                         + lowerBoundSize;

	/* Check memory requirements */
	/* Add one to error message to ceil */
	
	if (totalSize > (Size) maintenance_work_mem * 1024L)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("memory required is %zu MB, maintenance_work_mem is %d MB",
						totalSize / (1024 * 1024) + 1, maintenance_work_mem / 1024)));
     
	/* Ensure indexing does not overflow */
	if (numCenters * numCenters > INT_MAX)
		elog(ERROR, "Indexing overflow detected. Please report a bug.");

	/* Set support functions */
	procinfo = index_getprocinfo(index, 1, IVFFLAT_KMEANS_DISTANCE_PROC);
	normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_KMEANS_NORM_PROC);
	collation = index->rd_indcollation[0];


	///* Allocate space */
	///* Use float instead of double to save memory */
	//agg = palloc(aggSize);
	//centerCounts = palloc(centerCountsSize);
	//closestCenters = palloc(closestCentersSize);
	//lowerBound = palloc_extended(lowerBoundSize, MCXT_ALLOC_HUGE);

#ifdef IVFFLAT_MEMORY
	ShowMemoryUsage(MemoryContextGetParent(CurrentMemoryContext));
#endif
	/*Пока не отлажено*/
	//Initialize with kmeans++
	/* Pick initial centers */
	//InitCenters(index, samples, centers, lowerBound);
	//RandomCenters(index, centers, typeInfo); //LE

	/* Assign each x to its closest initial center c(x) = argmin d(x,c) */
	//for (int64 j = 0; j < numSamples; j++)
	//{
	//	float		minDistance = FLT_MAX;
	//	int			closestCenter = 0;

	//	/* Find closest center */
	//	for (int64 k = 0; k < numCenters; k++)
	//	{
	//		/* TODO Use Lemma 1 in k-means++ initialization */
	//		//float		distance = lowerBound[j * numCenters + k]; //LE comment
	//		//float distance = euclidean_distance(samples[i], centers[j], dim);
	//
	//		//LE get next center
	//		Datum		vec = PointerGetDatum(VectorArrayGet(centers, j));
	//		float distance = DatumGetFloat8(FunctionCall2Coll(procinfo, collation, vec, PointerGetDatum(VectorArrayGet(centers, k))));

	//		if (distance < minDistance)
	//		{
	//			minDistance = distance;
	//			closestCenter = k;
	//		}
	//	}
	//	//upperBound[j] = minDistance; //LE comment
	//	closestCenters[j] = closestCenter;
	//}

	///* Initialize new centers */
	//newCenters = VectorArrayInit(numCenters, dimensions, centers->itemsize);
	//newCenters->length = numCenters;

	///* It is debugging message */
	
	elog(INFO, "Simple K-means is running");			 
	/*
	int k = centers.length; //numSamples
	int dim = samples.dim;  //dimensions
	int *assignments = (int *)malloc(samples.length * sizeof(int));  //samplesize ?
	int changed;
	
	// Initialize assignments
	for (int i = 0; i < samples.length; i++)
	    {
	        assignments[i] = -1;
	    }
	
	do
	    {
	        changed = 0;
	
	        // Assignment step
	        for (int i = 0; i < samples.length; i++)
	        {
	            double min_dist = DBL_MAX;
	            int best_cluster = -1;
	
	            for (int j = 0; j < k; j++)
	            {
	                double dist = euclidean_distance(samples[i], centers[j], dim);
	                if (dist < min_dist)
	                {
	                    min_dist = dist;
	                    best_cluster = j;
	                }
	            }
	
	            if (assignments[i] != best_cluster)
	            {
	                assignments[i] = best_cluster;
	                changed = 1;
	            }
	        }
	
	        // Update step
	        update_centroids(samples, centers, assignments, k, dim);
	
	    } while (changed);
	
	    free(assignments);
	*/	
}

void UploadSamples(Relation index, VectorArray samples, const char *filename, int num_samples)
{
	char abs_path[PATH_MAX];
	char	   id_str[10];
	//char	   *ptr;

	elog(INFO, "UploadSamples is running");

	if (realpath(filename, abs_path)==NULL)
	{
        elog(ERROR,"Failed to get absolute path");
        exit(EXIT_FAILURE);		
	}
	elog(INFO,"Absolute path: %s\n", abs_path);
	
    FILE *file = fopen(abs_path, "w");
    if (!file)
    {
        elog(ERROR,"Failed to open file");
        exit(EXIT_FAILURE);
    }

	//buf = (char *) palloc(FLOAT_SHORTEST_DECIMAL_LEN * dim + 2);
	//ptr = buf;

    // Запись заголовка
    fprintf(file, "Index");
    for (int i = 0; i < samples->dim; i++)
    {
        fprintf(file, ",Feature%d", i + 1);
    }
    fprintf(file, "\n");

    // Запись данных
    for (int i = 0; i < num_samples && i < samples->length; i++)
    {
        //fprintf(file, "%d", index[i]); // TODO Запись индекса
		fprintf(file, "%d", i); // Отладка вместо индекса пока записываем счетчик
		//Получаем элемент из массива
		Vector* vec = PG_DETOAST_DATUM(PointerGetDatum(VectorArrayGet(samples, i)));
		//char	   *out = DatumGetPointer(DirectFunctionCall1(vector_out, PointerGetDatum(VectorArrayGet(samples, i))));
		//int		    dim = samples->dim;
        //for (int j = 0; j < dim; j++)
        //{
			//char *  elem = sample->items[j * samples->itemsize];
            //fprintf(file, ",%s", sample->x[i]); // Запись значений образцов
			//fprintf(file, ",%d", PG_DETOAST_DATUM(VectorArrayGet(samples, i)[0])); // Запись значений образцов
			//int sz = VectorArrayGet(samples, i)->Size
			//fprintf(file, ",%d", DatumGetInt32(vec)); // Запись значений образцов
			//fprintf(file, ",%d", VectorArrayGet(samples, i)[j]); // Запись значений образцов
			//fprintf(file, ",%s", VectorArrayGet(samples, 0)[0]); // Запись значений образцов
        //}
		//fprintf(file, ",%s",DatumGetPointer(DirectFunctionCall1(vector_out, vec)))
		snprintf(id_str, sizeof id_str, "%d", i);
		//PrintVector(id_str,vec); //Вывод отладочной информации
		fprintf(file, "%s", VectorToStr(vec));
        fprintf(file, "\n");
    }

    fclose(file);
}

void LoadCenters(Relation index, VectorArray centers, const IvfflatTypeInfo * typeInfo, const char *filename)
{
	char 		abs_path[PATH_MAX];
	char	   *ptr;
	char	   *buf;
	int 		i,j; 
	int			dimensions = centers->dim;
	int			numCenters = centers->maxlen;
	int 		sscanf_res = 0; //результат выполнения чтения из строки
	char       *id; //идентификатор образца из файла
	char       *fl_number; //число, считанное из файла в виде строки
	float       result;	   //результат парсинга строки в число типа float
	//FmgrInfo   *normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_KMEANS_NORM_PROC); //процедура нормализации
	//Oid			collation; //индекс
	VectorArray newCenters; //В этот массив первоначально загружаются векторы из файла
	float	   *x = (float *) palloc(sizeof(float) * dimensions); //элементы вектора
	char 		*line; //строка, считываемая из файла
	char 		*stopstring; //разделитель элементов в строке файла
	size_t len = 0; //используется при считывании строки
    ssize_t read;   //результат считывания строки
	
	line = (char *) palloc(DOUBLE_SHORTEST_DECIMAL_LEN * dimensions + DOUBLE_SHORTEST_DECIMAL_LEN + dimensions+2);
	id = (char*) palloc(DOUBLE_SHORTEST_DECIMAL_LEN);
	fl_number = (char*) palloc(DOUBLE_SHORTEST_DECIMAL_LEN);

	//Проверяем существование файла
	if (realpath(filename, abs_path)==NULL)
	{
        elog(ERROR,"Failed to get absolute path");
        exit(EXIT_FAILURE);		
	}
	elog(INFO,"Loading from file: %s\n", abs_path);
	
	//Открываем файл на чтение
    FILE *file = fopen(abs_path, "r");
    if (!file)
    {
        elog(ERROR,"Failed to open file");
        exit(EXIT_FAILURE);
    }

	//Проверка количества записей в файле
	//Подсчитываем количество строк
	i  = 0;
	while ((read = getline(&line, &len, file)) != -1) {
		//Вывод отладочной информации
        //elog(INFO,"Retrieved line of length %zu :\n", read);
        //elog(INFO,"%s", line);
		i++;
    };
	if ((i-1)!=(centers->length)){
		//Первая строка - это заголовок 
		elog(ERROR,"centers->length = %d, but retrieved %d vector lines from file",centers->length, i-1); //Вывод сообщения об ошибке
	};

	/* Initialize new centers */
	newCenters = VectorArrayInit(numCenters, dimensions, centers->itemsize);
	newCenters->length = numCenters;

#ifdef IVFFLAT_MEMORY
	ShowMemoryUsage(MemoryContextGetParent(CurrentMemoryContext));
#endif	

	newCenters->length = 0;
	//Отладочная информация
	//elog(INFO,"centers->length %d ", newCenters->length);
	//elog(INFO,"centers->maxlen %d ", newCenters->maxlen);

	rewind(file); //Устанавливаем указатель на начало файла
	//Считываем первую строку из файла, это строка заголовка
	read = getline(&line, &len, file);

	/* Fill from file data */
	while ((newCenters->length < newCenters->maxlen) && ((read = getline(&line, &len, file)) != -1))
	{	
		Pointer		center = VectorArrayGet(newCenters, newCenters->length);
		elog(INFO, "Считана строка %s",line);
		//удаляем символ новой строки
		ptr = strchr( line, '\n' );
    	if ( ptr ) *ptr = '\0';		
		//считываем идентификатор вектора
		sscanf_res = sscanf(line, "%[^'[']", id);
		if (sscanf_res!=1)
			{
				elog(ERROR,"Не удалось считать идентификатор элемента из строки %s",line);
			}
		else
			{
				//TODO что делать с идентификатором вектора?
				//отладочная информация
				//elog(INFO,"sscanf_res=%d, id=%s", sscanf_res, id);
			};
		
		for (i = 0; i < newCenters->dim; i++)
		{
			//позиционируемся на элементе
			if (i==0){ //разбираем первый элемент вектора	
					//Находим начало вектора данных (начинается и заканчивается символами [])		
					if ( (ptr=strchr( line, '[' )) != NULL ) {
        					ptr++; //нашли, //нашли, позиционируемся на первом символе
							buf = ptr;
    				} else {
						//не нашли, возможно не в том формате файл данных
						elog(ERROR,"Неверный формат файла, вектор данных должен быть заключен в квадратные скобки");
    				};
			}	
			else {
				//по разделителю (запятая) позиционируемся на следующее значение
				if ( (ptr=strchr( buf, ',' )) != NULL){
					ptr++; //нашли, позиционируемся на первом символе
					buf = ptr;
				}
				else {
						//не нашли, возможно не в том формате файл данных
						elog(ERROR,"Неверный формат файла, элементы вектора данных должен быть разделены запятыми");
    			};
			}
				
			//разбираем текущий элемент								
        	sscanf_res = sscanf( buf, "%[^,]", fl_number );
			//отладочная информация
			//elog(INFO,"разбираем строку %s", buf);
			//elog(INFO,"sscanf_res=%d, elem=%s",sscanf_res, fl_number);

			//Преобразуем считанное число в тип float
			//TODO обработка результата
      		result = strtof(fl_number, &stopstring);
			elog(INFO,"x[%d] = %f", i, result);
			//отладочная информация
			//elog(INFO,"Converted the %s value to the value %f", fl_number, result);

			//x[i] = (float) RandomDouble(); //отладка
			//Присваиваем новое значение элементу массива новых центроидов
			x[i] = result;
		}
		typeInfo->updateCenter(center, newCenters->dim, x);

		newCenters->length++;
		//отладочная информация
		//elog(INFO,"new centers->length %d ", newCenters->length);
	}

	//Отладка. Проверяем, как загрузили
	//Vector* vec = PG_DETOAST_DATUM(PointerGetDatum(VectorArrayGet(newCenters, 0)));
	//PrintVector("0",vec);

	//Update centers
	for (j = 0; j < numCenters; j++)
		VectorArraySet(centers, j, VectorArrayGet(newCenters, j));

	//TODO Нужна ли нормализация?	
	//if (normprocinfo != NULL)
	//	NormCenters(typeInfo, collation, centers);

	elog(INFO,"%d centers was loaded", centers->length);	

    fclose(file);
	free(line);
}


/*
 * Use Elkan for performance. This requires distance function to satisfy triangle inequality.
 *
 * We use L2 distance for L2 (not L2 squared like index scan)
 * and angular distance for inner product and cosine distance
 *
 * https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf
 */
static void
ElkanKmeans(Relation index, VectorArray samples, VectorArray centers, const IvfflatTypeInfo * typeInfo)
{
	FmgrInfo   *procinfo;
	FmgrInfo   *normprocinfo;
	Oid			collation;
	int			dimensions = centers->dim;
	int			numCenters = centers->maxlen;
	int			numSamples = samples->length;
	VectorArray newCenters;
	float	   *agg;
	int		   *centerCounts;
	int		   *closestCenters;
	float	   *lowerBound;
	float	   *upperBound;
	float	   *s;
	float	   *halfcdist;
	float	   *newcdist;

	/* Calculate allocation sizes */
	Size		samplesSize = VECTOR_ARRAY_SIZE(samples->maxlen, samples->itemsize);
	Size		centersSize = VECTOR_ARRAY_SIZE(centers->maxlen, centers->itemsize);
	Size		newCentersSize = VECTOR_ARRAY_SIZE(numCenters, centers->itemsize);
	Size		aggSize = sizeof(float) * (int64) numCenters * dimensions;
	Size		centerCountsSize = sizeof(int) * numCenters;
	Size		closestCentersSize = sizeof(int) * numSamples;
	Size		lowerBoundSize = sizeof(float) * numSamples * numCenters;
	Size		upperBoundSize = sizeof(float) * numSamples;
	Size		sSize = sizeof(float) * numCenters;
	Size		halfcdistSize = sizeof(float) * numCenters * numCenters;
	Size		newcdistSize = sizeof(float) * numCenters;

	/* Calculate total size */
	Size		totalSize = samplesSize + centersSize + newCentersSize + aggSize + centerCountsSize + closestCentersSize + lowerBoundSize + upperBoundSize + sSize + halfcdistSize + newcdistSize;

	/* Check memory requirements */
	/* Add one to error message to ceil */
	if (totalSize > (Size) maintenance_work_mem * 1024L)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("memory required is %zu MB, maintenance_work_mem is %d MB",
						totalSize / (1024 * 1024) + 1, maintenance_work_mem / 1024)));

	/* Ensure indexing does not overflow */
	if (numCenters * numCenters > INT_MAX)
		elog(ERROR, "Indexing overflow detected. Please report a bug.");

	/* Set support functions */
	procinfo = index_getprocinfo(index, 1, IVFFLAT_KMEANS_DISTANCE_PROC);
	normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_KMEANS_NORM_PROC);
	collation = index->rd_indcollation[0];

	/* Allocate space */
	/* Use float instead of double to save memory */
	agg = palloc(aggSize);
	centerCounts = palloc(centerCountsSize);
	closestCenters = palloc(closestCentersSize);
	lowerBound = palloc_extended(lowerBoundSize, MCXT_ALLOC_HUGE);
	upperBound = palloc(upperBoundSize);
	s = palloc(sSize);
	halfcdist = palloc_extended(halfcdistSize, MCXT_ALLOC_HUGE);
	newcdist = palloc(newcdistSize);

	/* Initialize new centers */
	newCenters = VectorArrayInit(numCenters, dimensions, centers->itemsize);
	newCenters->length = numCenters;

#ifdef IVFFLAT_MEMORY
	ShowMemoryUsage(MemoryContextGetParent(CurrentMemoryContext));
#endif

	/* Pick initial centers */
	InitCenters(index, samples, centers, lowerBound);

	/* Assign each x to its closest initial center c(x) = argmin d(x,c) */
	for (int64 j = 0; j < numSamples; j++)
	{
		float		minDistance = FLT_MAX;
		int			closestCenter = 0;

		/* Find closest center */
		for (int64 k = 0; k < numCenters; k++)
		{
			/* TODO Use Lemma 1 in k-means++ initialization */
			float		distance = lowerBound[j * numCenters + k];

			if (distance < minDistance)
			{
				minDistance = distance;
				closestCenter = k;
			}
		}

		upperBound[j] = minDistance;
		closestCenters[j] = closestCenter;
	}

	/* Give 500 iterations to converge */
	for (int iteration = 0; iteration < 500; iteration++)
	{
		int			changes = 0;
		bool		rjreset;

		/* Can take a while, so ensure we can interrupt */
		CHECK_FOR_INTERRUPTS();

		/* Step 1: For all centers, compute distance */
		for (int64 j = 0; j < numCenters; j++)
		{
			Datum		vec = PointerGetDatum(VectorArrayGet(centers, j));

			for (int64 k = j + 1; k < numCenters; k++)
			{
				float		distance = 0.5 * DatumGetFloat8(FunctionCall2Coll(procinfo, collation, vec, PointerGetDatum(VectorArrayGet(centers, k))));

				halfcdist[j * numCenters + k] = distance;
				halfcdist[k * numCenters + j] = distance;
			}
		}

		/* For all centers c, compute s(c) */
		for (int64 j = 0; j < numCenters; j++)
		{
			float		minDistance = FLT_MAX;

			for (int64 k = 0; k < numCenters; k++)
			{
				float		distance;

				if (j == k)
					continue;

				distance = halfcdist[j * numCenters + k];
				if (distance < minDistance)
					minDistance = distance;
			}

			s[j] = minDistance;
		}

		rjreset = iteration != 0;

		for (int64 j = 0; j < numSamples; j++)
		{
			bool		rj;

			/* Step 2: Identify all points x such that u(x) <= s(c(x)) */
			if (upperBound[j] <= s[closestCenters[j]])
				continue;

			rj = rjreset;

			for (int64 k = 0; k < numCenters; k++)
			{
				Datum		vec;
				float		dxcx;

				/* Step 3: For all remaining points x and centers c */
				if (k == closestCenters[j])
					continue;

				if (upperBound[j] <= lowerBound[j * numCenters + k])
					continue;

				if (upperBound[j] <= halfcdist[closestCenters[j] * numCenters + k])
					continue;

				vec = PointerGetDatum(VectorArrayGet(samples, j));

				/* Step 3a */
				if (rj)
				{
					dxcx = DatumGetFloat8(FunctionCall2Coll(procinfo, collation, vec, PointerGetDatum(VectorArrayGet(centers, closestCenters[j]))));

					/* d(x,c(x)) computed, which is a form of d(x,c) */
					lowerBound[j * numCenters + closestCenters[j]] = dxcx;
					upperBound[j] = dxcx;

					rj = false;
				}
				else
					dxcx = upperBound[j];

				/* Step 3b */
				if (dxcx > lowerBound[j * numCenters + k] || dxcx > halfcdist[closestCenters[j] * numCenters + k])
				{
					float		dxc = DatumGetFloat8(FunctionCall2Coll(procinfo, collation, vec, PointerGetDatum(VectorArrayGet(centers, k))));

					/* d(x,c) calculated */
					lowerBound[j * numCenters + k] = dxc;

					if (dxc < dxcx)
					{
						closestCenters[j] = k;

						/* c(x) changed */
						upperBound[j] = dxc;

						changes++;
					}
				}
			}
		}

		/* Step 4: For each center c, let m(c) be mean of all points assigned */
		ComputeNewCenters(samples, agg, newCenters, centerCounts, closestCenters, normprocinfo, collation, typeInfo);

		/* Step 5 */
		for (int j = 0; j < numCenters; j++)
			newcdist[j] = DatumGetFloat8(FunctionCall2Coll(procinfo, collation, PointerGetDatum(VectorArrayGet(centers, j)), PointerGetDatum(VectorArrayGet(newCenters, j))));

		for (int64 j = 0; j < numSamples; j++)
		{
			for (int64 k = 0; k < numCenters; k++)
			{
				float		distance = lowerBound[j * numCenters + k] - newcdist[k];

				if (distance < 0)
					distance = 0;

				lowerBound[j * numCenters + k] = distance;
			}
		}

		/* Step 6 */
		/* We reset r(x) before Step 3 in the next iteration */
		for (int j = 0; j < numSamples; j++)
			upperBound[j] += newcdist[closestCenters[j]];

		/* Step 7 */
		for (int j = 0; j < numCenters; j++)
			VectorArraySet(centers, j, VectorArrayGet(newCenters, j));

		if (changes == 0 && iteration != 0)
			break;
	}
}

/*
 * Ensure no NaN or infinite values
 */
static void
CheckElements(VectorArray centers, const IvfflatTypeInfo * typeInfo)
{
	float	   *scratch = palloc(sizeof(float) * centers->dim);

	for (int i = 0; i < centers->length; i++)
	{
		for (int j = 0; j < centers->dim; j++)
			scratch[j] = 0;

		/* /fp:fast may not propagate NaN with MSVC, but that's alright */
		typeInfo->sumCenter(VectorArrayGet(centers, i), scratch);

		for (int j = 0; j < centers->dim; j++)
		{
			if (isnan(scratch[j]))
				elog(ERROR, "NaN detected. Please report a bug.");

			if (isinf(scratch[j]))
				elog(ERROR, "Infinite value detected. Please report a bug.");
		}
	}
}

/*
 * Ensure no zero vectors for cosine distance
 */
static void
CheckNorms(VectorArray centers, Relation index)
{
	/* Check NORM_PROC instead of KMEANS_NORM_PROC */
	FmgrInfo   *normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_NORM_PROC);
	Oid			collation = index->rd_indcollation[0];

	if (normprocinfo == NULL)
		return;

	for (int i = 0; i < centers->length; i++)
	{
		double		norm = DatumGetFloat8(FunctionCall1Coll(normprocinfo, collation, PointerGetDatum(VectorArrayGet(centers, i))));

		if (norm == 0)
			elog(ERROR, "Zero norm detected. Please report a bug.");
	}
}

/*
 * Detect issues with centers
 */
static void
CheckCenters(Relation index, VectorArray centers, const IvfflatTypeInfo * typeInfo)
{
	if (centers->length != centers->maxlen)
		elog(ERROR, "Not enough centers. Please report a bug.");

	CheckElements(centers, typeInfo);
	CheckNorms(centers, index);
}

/*
 * Perform naive k-means centering
 * We use spherical k-means for inner product and cosine
 */
void
IvfflatKmeans(Relation index, VectorArray samples, VectorArray centers, const IvfflatTypeInfo * typeInfo)
{
	MemoryContext kmeansCtx = AllocSetContextCreate(CurrentMemoryContext,
													"Ivfflat kmeans temporary context",
													ALLOCSET_DEFAULT_SIZES);
	MemoryContext oldCtx = MemoryContextSwitchTo(kmeansCtx);

	if (samples->length == 0)
		RandomCenters(index, centers, typeInfo);
	else
		{
		ElkanKmeans(index, samples, centers, typeInfo); //LE comment
		
		//Тестовая выгрузка образцов во внешний файл
		UploadSamples(index, centers,"/mnt/c/Users/sept_/huawei/pgvector/samples.csv",100);
		//Тестовая загрузка образцов из внешнего файла
		LoadCenters(index, centers, typeInfo, "/mnt/c/Users/sept_/huawei/pgvector/samples.csv");
		//TODO LE change to another k-means algorithm	
		//SimpleKmeans(index, samples, centers, typeInfo);	
		}

	CheckCenters(index, centers, typeInfo);

	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(kmeansCtx);
}
