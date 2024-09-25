#include "postgres.h"
#include "fmgr.h"
#include "vector.h"
#include "ivfflat.h"

//Функция загрузки элементов из файла .CSV в массив centers
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(load_centers_from_csv);
Datum 
load_centers_from_csv(PG_FUNCTION_ARGS)
{
    //Получаем полный путь к файлу в виде строки
    VarChar *arg_a = PG_GETARG_VARCHAR_P(0);
    //Вызываем функцию загрузки центров из файла .csv
    LoadCenters(index, centers, typeInfo, "/mnt/c/Users/sept_/huawei/pgvector/samples.csv");
    //Успешное завершение
    PG_RETURN_INT32(0);
}
