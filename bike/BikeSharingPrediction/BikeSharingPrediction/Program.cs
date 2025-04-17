using BikeSharingPrediction.Models.Input;
using BikeSharingPrediction.Services.Interfaces;
using BikeSharingPrediction.Services;
using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using System.Globalization;

namespace BikeSharingPrediction
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Установка культуры для корректного парсинга чисел
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

            // Конфигурация
            var configuration = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json")
                .Build();

            // Инициализация MLContext
            var mlContext = new MLContext(seed: configuration.GetValue<int>("ModelSettings:Seed", 42));

            // Инициализация сервисов
            var dataService = new DataService(mlContext, configuration);
            var modelService = new ModelService(mlContext, configuration);

            try
            {
                Console.WriteLine("Loading data...");
                var fullData = dataService.LoadData();

                Console.WriteLine("Splitting data...");
                var splitData = dataService.SplitData(fullData);

                Console.WriteLine("Training model...");
                var model = modelService.TrainModel(splitData.TrainSet);

                Console.WriteLine("Evaluating model...");
                var metrics = modelService.EvaluateModel(model, splitData.TestSet);

                Console.WriteLine($"Model metrics:");
                Console.WriteLine($"  Accuracy: {metrics.Accuracy:P2}");
                Console.WriteLine($"  AUC: {metrics.AreaUnderRocCurve:P2}");
                Console.WriteLine($"  F1-Score: {metrics.F1Score:P2}");

                // Пример предсказания
                var predictionEngine = modelService.CreatePredictionEngine(model);

                var sample = new BikeRentalData
                {
                    Season = 3,
                    Month = 7,
                    Hour = 17,
                    Holiday = 0,
                    Weekday = 3,
                    WorkingDay = 1,
                    WeatherCondition = 1,
                    Temperature = 25.0f,
                    Humidity = 65.0f,
                    Windspeed = 10.0f
                };

                var prediction = predictionEngine.Predict(sample);
                Console.WriteLine($"\nPrediction for sample:");
                Console.WriteLine($"  Type: {(prediction.PredictedRentalType ? "Long" : "Short")}-term");
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
