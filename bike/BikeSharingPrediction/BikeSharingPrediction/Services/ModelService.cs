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

        public IEstimator<ITransformer> BuildDataPipeline()
        {
            return _mlContext.Transforms
                .Conversion.MapValueToKey("LabelKey", "Label")
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding(
                    "SeasonEncoded", "Season"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding(
                    "WeatherEncoded", "WeatherCondition"))
                .Append(_mlContext.Transforms.Concatenate(
                    "Features",
                    "SeasonEncoded", "WeatherEncoded",
                    "Hour", "Temperature", "Humidity", "Windspeed",
                    "Holiday", "WorkingDay"))
                .Append(_mlContext.Transforms.NormalizeMinMax("Features"));
        }

        public ITransformer TrainModel(IDataView trainData)
        {
            var pipeline = BuildDataPipeline();
            return pipeline.Fit(trainData);
        }

        public CalibratedBinaryClassificationMetrics EvaluateModel(
            ITransformer model, IDataView testData)
        {
            var predictions = model.Transform(testData);
            return _mlContext.BinaryClassification.Evaluate(
                predictions,
                labelColumnName: "LabelKey");
        }

        public void SaveModel(ITransformer model, DataViewSchema schema)
        {
            var modelDir = Path.GetDirectoryName(_modelPath);
            if (!Directory.Exists(modelDir))
                Directory.CreateDirectory(modelDir);

            _mlContext.Model.Save(model, schema, _modelPath);
        }

        public PredictionEngine<BikeRentalData, BikeRentalPrediction> CreatePredictionEngine(
            ITransformer model)
        {
            return _mlContext.Model.CreatePredictionEngine<
                BikeRentalData, BikeRentalPrediction>(model);
        }
    }
}

