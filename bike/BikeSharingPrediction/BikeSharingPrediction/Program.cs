using BikeSharingPrediction.Models.Input;
using BikeSharingPrediction.Services.Interfaces;
using BikeSharingPrediction.Services;
using Microsoft.Extensions.Configuration;
using Microsoft.ML;

namespace BikeSharingPrediction
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var configuration = new ConfigurationBuilder()
     .SetBasePath(Directory.GetCurrentDirectory())
     .AddJsonFile("Config/appsettings.json")
     .Build();

            // Инициализация MLContext с seed из конфига
            var mlContext = new MLContext(
                seed: int.Parse(configuration["ModelSettings:Seed"]));

            // Инициализация сервисов
            IDataService dataService = new DataService(mlContext, configuration);
            IModelService modelService = new ModelService(mlContext, configuration);

            try
            {
                // 1. Загрузка данных
                Console.WriteLine("Loading data...");
                var fullData = dataService.LoadData();

                // 2. Разделение данных
                Console.WriteLine("Splitting data...");
                var splitData = dataService.SplitData(fullData);

                // 3. Построение и обучение модели
                Console.WriteLine("Training model...");
                var model = modelService.TrainModel(splitData.TrainSet);

                // 4. Оценка модели
                Console.WriteLine("Evaluating model...");
                var metrics = modelService.EvaluateModel(model, splitData.TestSet);
                Console.WriteLine($"Model metrics:\n" +
                                  $"  Accuracy: {metrics.Accuracy:P2}\n" +
                                  $"  AUC: {metrics.AreaUnderRocCurve:P2}\n" +
                                  $"  F1Score: {metrics.F1Score:P2}");

                // 5. Сохранение модели
                Console.WriteLine("Saving model...");
                modelService.SaveModel(model, fullData.Schema);

                // 6. Пример предсказания
                var predictionEngine = modelService.CreatePredictionEngine(model);

                var sampleData = new BikeRentalData
                {
                    Season = 3,      // Лето
                    Month = 7,       // Июль
                    Hour = 17,       // 5 PM
                    Holiday = 0,     // Не выходной
                    Weekday = 3,     // Среда
                    WorkingDay = 1,  // Рабочий день
                    WeatherCondition = 1, // Ясно
                    Temperature = 25.0f,
                    Humidity = 65.0f,
                    Windspeed = 10.0f
                };

                var prediction = predictionEngine.Predict(sampleData);
                Console.WriteLine("\nSample prediction:");
                Console.WriteLine($"  Predicted: {(prediction.PredictedRentalType ? "Long" : "Short")} term rental");
                Console.WriteLine($"  Probability: {prediction.Probability:P2}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                if (ex.InnerException != null)
                    Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
            }

            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();
        }
    }
}
