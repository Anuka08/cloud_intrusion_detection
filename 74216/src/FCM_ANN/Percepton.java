package FCM_ANN;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Percepton {

    public static double LEARNING_RATE = 0.05;
    public static ArrayList<Double> INITIAL_WEIGHTS = new ArrayList();

    public static void initialize_data(ArrayList<ArrayList<Double>> data) {
        
        //weight
        for (int i = 0; i < data.get(0).size(); i++) {    
            INITIAL_WEIGHTS.add(Math.random());
        }
    }

    public double calculateWeightedSum(ArrayList<Double> data, ArrayList<Double> weights) {
        double weightedSum = 0;
        for (int x = 0; x < data.size(); x++) {
            weightedSum += data.get(x) * weights.get(x);
        }
        return weightedSum;
    }

    public double applyActivationFunction(double weightedSum, double clas , int p, int x, ArrayList<ArrayList<Double>> data, ArrayList<Double> data_clas) {
        double result = 0;
        if (weightedSum > 1 && x < p)
            result = clas;
        else {
            int c = (int)(Math.random()*data.size());
            result = (int)(Math.random()*2);
        }
        return result;
    }

    public double[] adjustWeights(int[] data, double[] weights, double error) {
        double[] adjustedWeights = new double[weights.length];
        for (int x = 0; x < weights.length; x++) {
            adjustedWeights[x] = LEARNING_RATE * error * data[x] + weights[x];
        }
        return adjustedWeights;
    }
    
    public static ArrayList<ArrayList<Double>> normalizing(ArrayList<ArrayList<Double>> input, int range) {
        ArrayList<ArrayList<Double>> normalizeddata = new ArrayList<>();
        ArrayList<ArrayList<Double>> transposedata = new ArrayList<>();   //to normalize -- consider column
        transposedata = transpose(input);

        //normalize
        for (int i = 0; i < transposedata.size(); i++) {
            ArrayList<Double> tem = new ArrayList<>();
            tem = transposedata.get(i);
            double min = Collections.min(tem);
            double max = Collections.max(tem);
            double del = (max - min) / range;   //range = 2 (0 & 1)

            //levels
            ArrayList<Double> levels = new ArrayList();
            for (int k = 1; k <= range; k++) {       //levels of value (size -- range - 1)
                levels.add((min + (k * del)));
            }

            ArrayList<Double> normtem = new ArrayList<>();
            for (int j = 0; j < transposedata.get(i).size(); j++) {
                if(transposedata.get(i).get(j) < levels.get(range-2)) {      //data value < before last value in levels
                    double m = 0;
                    int k1 = 0;
                    while (k1 < levels.size()) {
                        if(transposedata.get(i).get(j) < levels.get(k1)) {
                            m = k1;         //check the data value is  < which value in levels
                            k1 = levels.size();
                        }
                        k1++;
                    }
                    normtem.add(m);         //add that level location as normalized data
                }
                else
                    normtem.add((double)(range-1));     //else max range is added
            } 
            normalizeddata.add(normtem);
        }
        normalizeddata = transpose(normalizeddata);
        return normalizeddata;
    }
    
    public static ArrayList<ArrayList<Double>> transpose(ArrayList<ArrayList<Double>> table) {
        ArrayList<ArrayList<Double>> ret = new ArrayList<ArrayList<Double>>();
        final int N = table.get(0).size();
        for (int i = 0; i < N; i++) {
            ArrayList<Double> col = new ArrayList<Double>();
            for (List<Double> row : table) {
                col.add(row.get(i));
            }
            ret.add(col);
        }
        return ret;
    }
}