package Proposed_SFDO_DRNN;

import java.util.ArrayList;
import java.util.Random;

public class DRNN {
    
    public static double acc, dr, fpr;
    
    public static ArrayList<Double> callmain(ArrayList<ArrayList<Double>> Feature_data, ArrayList<Double> clas, ArrayList<Double> opt_w) throws Exception {
        ArrayList<Double> metrics = new ArrayList();
        Random r = new Random(1234);
        DistractedSequenceRecall task = new DistractedSequenceRecall(r);

        int cell_blocks = 15;
        int tp = 225 + (Code.Run.tp/2);
        RNN drnn = new RNN(r, task.GetObservationDimension(), task.GetActionDimension(), cell_blocks, opt_w);

        for (int i = 0; i < tp; i++) {
            task.EvaluateFitnessSupervised(drnn);
        }
        metrics.add(acc); metrics.add(dr); metrics.add(fpr);
        
        return metrics;     // return the metrics from DRNN
    }
}