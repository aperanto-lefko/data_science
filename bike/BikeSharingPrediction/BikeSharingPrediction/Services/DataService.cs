using BikeSharingPrediction.Models.Input;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using BikeSharingPrediction.Services.Interfaces;

namespace BikeSharingPrediction.Services
{
    internal class DataService : IDataService
    {
        private readonly MLContext _mlContext;
        private readonly string _dataPath;
        private readonly double _testFraction;

        public DataService(MLContext mlContext, IConfiguration config)
        {
            _mlContext = mlContext;
            _dataPath = Path.Combine(Directory.GetCurrentDirectory(),
                                   config["DataSettings:TrainingDataPath"]);
            _testFraction = double.Parse(config["ModelSettings:TestFraction"]);

            if (!File.Exists(_dataPath))
                throw new FileNotFoundException($"CSV file not found at: {_dataPath}");
        }

        public IDataView LoadData()
        {
            return _mlContext.Data.LoadFromTextFile<BikeRentalData>(
                _dataPath,
                hasHeader: true,
                separatorChar: ',');
        }

        public DataOperationsCatalog.TrainTestData SplitData(IDataView data)
        {
            return _mlContext.Data.TrainTestSplit(data, testFraction: _testFraction);
        }
    }
}

