using BikeSharingPrediction.Models.Input;
using BikeSharingPrediction.Models.Output;
using Microsoft.ML.Data;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BikeSharingPrediction.Services.Interfaces;
using Microsoft.Extensions.Configuration;
using Microsoft.ML.Trainers.LightGbm;

namespace BikeSharingPrediction.Services
{
    internal class ModelService : IModelService
    {
       
        private readonly MLContext _mlContext;
        private readonly string _modelPath;

        public ModelService(MLContext mlContext, IConfiguration config)
        {
            _mlContext = mlContext;
            _modelPath = Path.Combine(Directory.GetCurrentDirectory(),
                                    config["DataSettings:ModelPath"]);
        }

        public ITransformer TrainModel(IDataView trainData)
        {
            var pipeline = _mlContext.Transforms
                .Conversion.MapValueToKey("LabelKey", "Label")
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding(
                    outputColumnName: "SeasonEncoded",
                    inputColumnName: nameof(BikeRentalData.Season)))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding(
                    outputColumnName: "WeatherEncoded",
                    inputColumnName: nameof(BikeRentalData.WeatherCondition)))
                .Append(_mlContext.Transforms.Concatenate(
                    outputColumnName: "Features",
                    "SeasonEncoded", "WeatherEncoded",
                    nameof(BikeRentalData.Hour),
                    nameof(BikeRentalData.Temperature),
                    nameof(BikeRentalData.Humidity),
                    nameof(BikeRentalData.Windspeed),
                    nameof(BikeRentalData.Holiday),
                    nameof(BikeRentalData.WorkingDay)))
                .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
                .AppendCacheCheckpoint(_mlContext);

            var trainingPipeline = pipeline
                .Append(_mlContext.BinaryClassification.Trainers.LightGbm(new LightGbmBinaryTrainer.Options
                {
                    LabelColumnName = "LabelKey",
                    FeatureColumnName = "Features",
                    NumberOfLeaves = 31,
                    MinimumExampleCountPerLeaf = 10,
                    LearningRate = 0.1,
                    NumberOfIterations = 100,
                    Booster = new GradientBooster.Options { L2Regularization = 0.5 }
                }));

            return trainingPipeline.Fit(trainData);
        }

        public CalibratedBinaryClassificationMetrics EvaluateModel(ITransformer model, IDataView testData)
        {
            var predictions = model.Transform(testData);
            return _mlContext.BinaryClassification.Evaluate(
                data: predictions,
                labelColumnName: "LabelKey",
                scoreColumnName: "Score");
        }

        public PredictionEngine<BikeRentalData, BikeRentalPrediction> CreatePredictionEngine(ITransformer model)
        {
            return _mlContext.Model.CreatePredictionEngine<BikeRentalData, BikeRentalPrediction>(model);
        }
    }
}

