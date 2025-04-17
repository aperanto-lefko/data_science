using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BikeSharingPrediction.Models.Input
{
    internal class BikeRentalData
    {
        [LoadColumn(0)] public float Season { get; set; }
        [LoadColumn(1)] public float Month { get; set; }
        [LoadColumn(2)] public float Hour { get; set; }
        [LoadColumn(3)] public float Holiday { get; set; }
        [LoadColumn(4)] public float Weekday { get; set; }
        [LoadColumn(5)] public float WorkingDay { get; set; }
        [LoadColumn(6)] public float WeatherCondition { get; set; }
        [LoadColumn(7)] public float Temperature { get; set; }
        [LoadColumn(8)] public float Humidity { get; set; }
        [LoadColumn(9)] public float Windspeed { get; set; }
        [LoadColumn(10), ColumnName("Label")]
        public bool RentalType { get; set; }
    }
}
