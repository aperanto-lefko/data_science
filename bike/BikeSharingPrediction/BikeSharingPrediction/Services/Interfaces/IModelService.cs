using BikeSharingPrediction.Models.Input;
using BikeSharingPrediction.Models.Output;
using Microsoft.ML.Data;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BikeSharingPrediction.Services.Interfaces
{
    internal interface IModelService
    {
        ITransformer TrainModel(IDataView trainData);
        CalibratedBinaryClassificationMetrics EvaluateModel(ITransformer model, IDataView testData);
        PredictionEngine<BikeRentalData, BikeRentalPrediction> CreatePredictionEngine(ITransformer model);
    }
    }

