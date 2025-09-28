package Proposed_SFDO_DRNN;

import static Proposed_SFDO_DRNN.SailFish_update.acc;
import static Proposed_SFDO_DRNN.SailFish_update.dr;
import static Proposed_SFDO_DRNN.SailFish_update.fpr;

import java.util.ArrayList;

public class SFDO_fitness {
    
    public static ArrayList<Double> func(ArrayList<ArrayList<Double>> solution, ArrayList<ArrayList<Double>> feature, ArrayList<Double> clas) throws Exception {
        
        ArrayList<Double> fit = new ArrayList();
        for (int i = 0; i <solution.size(); i++) {
            // each soln calculate fitness
            ArrayList<Double> fit_metrics = Proposed_SFDO_DRNN.DRNN.callmain(feature, clas, solution.get(i));
            fit.add(1.0 - fit_metrics.get(0));    // loss(fitness)
            acc.add(fit_metrics.get(0)); dr.add(fit_metrics.get(1)); fpr.add(fit_metrics.get(2));    // accuracy, dr, fpr
        }
        return fit;
    }
}
