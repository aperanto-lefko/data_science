using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BikeSharingPrediction.Services.Interfaces
{
    internal interface IDataService
    {
        IDataView LoadData();
        DataOperationsCatalog.TrainTestData SplitData(IDataView data);
    }
}
