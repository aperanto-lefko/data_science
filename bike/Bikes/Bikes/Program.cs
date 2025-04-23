using System.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace Bikes
{
    internal class Program
    {
        // Путь к файлам данных
        private static string _dataPath = "bike_sharing.csv";
        private static string _modelPath = "BikeRentalModel.zip";

        // Класс для описания структуры входных данных
        public class BikeRentalData
        {
            // Каждый атрибут соответствует столбцу в CSV-файле
            [LoadColumn(0)]
            public float Season { get; set; }

            [LoadColumn(1)]
            public float Month { get; set; }

            [LoadColumn(2)]
            public float Hour { get; set; }

            [LoadColumn(3)]
            public float Holiday { get; set; }

            [LoadColumn(4)]
            public float Weekday { get; set; }

            [LoadColumn(5)]
            public float WorkingDay { get; set; }

            [LoadColumn(6)]
            public float WeatherCondition { get; set; }

            [LoadColumn(7)]
            public float Temperature { get; set; }

            [LoadColumn(8)]
            public float Humidity { get; set; }

            [LoadColumn(9)]
            public float Windspeed { get; set; }

            [LoadColumn(10)]
            public bool RentalType { get; set; } // 0 = краткосрочная, 1 = долгосрочная
        }
        // Класс для хранения результатов предсказания
        public class RentalTypePrediction
        {
            [ColumnName("PredictedLabel")]
            public bool PredictedRentalType { get; set; }// Само предсказание

            public float Probability { get; set; }// Вероятность предсказания

            public float Score { get; set; }// "Сырой" score от модели
        }

        static void Main(string[] args)
        {
            Console.WriteLine("Предсказание типа аренды велосипеда с использованием ML.NET");

            // 1. Создание ML.NET контекста Инициализация ML-контекста (как "движка" ML.NET)
            var mlContext = new MLContext(seed: 0);

            // 2. Загрузка данных
            Console.WriteLine("Загрузка данных...");
            IDataView dataView = mlContext.Data.LoadFromTextFile<BikeRentalData>(
                path: _dataPath,
                hasHeader: true, // Первая строка - заголовки
                separatorChar: ','); 

            // 3. Разделение данных на обучающую и тестовую выборки (80%/20%)
            Console.WriteLine("Разделение данных на обучающую и тестовую выборки...");
            var trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var trainData = trainTestSplit.TrainSet;
            var testData = trainTestSplit.TestSet;

            // 4. Создание пайплайна обработки данных
            Console.WriteLine("Создание пайплайна обработки данных...");
            // Объединяем все признаки в один вектор "Features"
            var dataPipeline = mlContext.Transforms
    .Concatenate(
        "Features",
        nameof(BikeRentalData.Season),
        nameof(BikeRentalData.Month),
        nameof(BikeRentalData.Hour),
        nameof(BikeRentalData.Holiday),
        nameof(BikeRentalData.Weekday),
        nameof(BikeRentalData.WorkingDay),
        nameof(BikeRentalData.WeatherCondition),
        nameof(BikeRentalData.Temperature),
        nameof(BikeRentalData.Humidity),
        nameof(BikeRentalData.Windspeed))
    // Нормализуем числовые признаки (приводим к диапазону 0-1)
    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
    // Преобразуем целевую переменную в Boolean
    .Append(mlContext.Transforms.Conversion.ConvertType(
        outputColumnName: "Label",
        inputColumnName: nameof(BikeRentalData.RentalType),
        outputKind: DataKind.Boolean));

            // 5. Обучение моделей и выбор лучшей
            Console.WriteLine("Обучение моделей...");

            // Создаем и обучаем несколько моделей
            // FastTree (деревья решений)
            var fastTreeModel = TrainFastTree(mlContext, dataPipeline, trainData);
            // LightGBM (градиентный бустинг)
            var lightGbmModel = TrainLightGbm(mlContext, dataPipeline, trainData);
            // Логистическая регрессия
            var logisticRegressionModel = TrainLogisticRegression(mlContext, dataPipeline, trainData);

            // Оцениваем модели
            Console.WriteLine("\nОценка качества моделей:");
            EvaluateModel(mlContext, fastTreeModel, testData, "FastTree");
            EvaluateModel(mlContext, lightGbmModel, testData, "LightGBM");
            EvaluateModel(mlContext, logisticRegressionModel, testData, "Logistic Regression");

          
            Console.WriteLine("\nВыбираем модель LightGBM как лучшую...");
            var bestModel = lightGbmModel;

            // Сохраняем модель
            Console.WriteLine("Сохранение модели...");
            mlContext.Model.Save(bestModel, trainData.Schema, _modelPath);

            // 7. Выполнение предсказаний
            Console.WriteLine("\nВыполнение предсказаний на новых примерах...");
            var predictionEngine = mlContext.Model.CreatePredictionEngine<BikeRentalData, RentalTypePrediction>(bestModel);

            // Пример 1: Краткосрочная аренда (ожидается 0)
            var example1 = new BikeRentalData
            {
                Season = 3, 
                Month = 7,  
                Hour = 8,   
                Holiday = 0,
                Weekday = 2, 
                WorkingDay = 1,
                WeatherCondition = 1, 
                Temperature = 25.0f,
                Humidity = 60.0f,
                Windspeed = 10.0f
            };

            var prediction1 = predictionEngine.Predict(example1);
            Console.WriteLine($"Пример 1: Прогнозируемый тип аренды: {(prediction1.PredictedRentalType ? "Долгосрочная" : "Краткосрочная")} " +
                              $"(Вероятность: {prediction1.Probability:P2})");

            // Пример 2: Долгосрочная аренда (ожидается 1)
            var example2 = new BikeRentalData
            {
                Season = 1, 
                Month = 1,  
                Hour = 14,  
                Holiday = 0,
                Weekday = 3, 
                WorkingDay = 1,
                WeatherCondition = 2,
                Temperature = 2.0f,
                Humidity = 85.0f,
                Windspeed = 5.0f
            };

            var prediction2 = predictionEngine.Predict(example2);
            Console.WriteLine($"Пример 2: Прогнозируемый тип аренды: {(prediction2.PredictedRentalType ? "Долгосрочная" : "Краткосрочная")} " +
                              $"(Вероятность: {prediction2.Probability:P2})");

            Console.WriteLine("\nНажмите любую клавишу для завершения...");
            Console.ReadKey();
        }

        private static ITransformer TrainFastTree(MLContext mlContext, IEstimator<ITransformer> pipeline, IDataView trainData)
        {
            Console.WriteLine("Обучение модели FastTree...");
            var trainingPipeline = pipeline.Append(
                mlContext.BinaryClassification.Trainers.FastTree(
                    labelColumnName: "Label",
                    featureColumnName: "Features",
                    numberOfLeaves: 50, // Параметры дерева
                    numberOfTrees: 100,
                    learningRate: 0.1));

            return trainingPipeline.Fit(trainData); // Обучение на данных
        }

        private static ITransformer TrainLightGbm(MLContext mlContext, IEstimator<ITransformer> pipeline, IDataView trainData)
        {
            Console.WriteLine("Обучение модели LightGBM...");
            var trainingPipeline = pipeline.Append(mlContext.BinaryClassification.Trainers.LightGbm(
                numberOfLeaves: 50,
                numberOfIterations: 100,
                learningRate: 0.1));

            return trainingPipeline.Fit(trainData);
        }

        private static ITransformer TrainLogisticRegression(MLContext mlContext, IEstimator<ITransformer> pipeline, IDataView trainData)
        {
            Console.WriteLine("Обучение модели Logistic Regression...");

            var options = new Microsoft.ML.Trainers.LbfgsLogisticRegressionBinaryTrainer.Options()
            {
                MaximumNumberOfIterations = 100,
                LabelColumnName = "Label",
                FeatureColumnName = "Features"
            };

            var trainingPipeline = pipeline.Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(options));

            return trainingPipeline.Fit(trainData);
        }

        private static void EvaluateModel(MLContext mlContext, ITransformer model, IDataView testData, string modelName)
        {
            Console.WriteLine($"\nОценка модели {modelName}...");
            var predictions = model.Transform(testData);

            var metrics = mlContext.BinaryClassification.Evaluate(
                data: predictions,
                labelColumnName: "Label",
                scoreColumnName: "Score");

            Console.WriteLine($"  AUC: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"  F1 Score: {metrics.F1Score:P2}");
            Console.WriteLine($"  Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"  Log Loss: {metrics.LogLoss:F4}");
        }
    }
}
    
