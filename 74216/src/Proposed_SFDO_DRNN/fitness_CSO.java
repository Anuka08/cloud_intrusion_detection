package Proposed_SFDO_DRNN;

import java.util.ArrayList;
import static Code.Run.VM_Migration;
import static Proposed_SFDO_DRNN.run.VM_Migration_update;
import static Proposed_SFDO_DRNN.run.st;
import static Proposed_SFDO_DRNN.run.p;
import static Proposed_SFDO_DRNN.run.load;
import static Proposed_SFDO_DRNN.run.resource_utilization;
import static Proposed_SFDO_DRNN.run.resource_utilization_C;
import static Proposed_SFDO_DRNN.run.resource_availability;
import static Proposed_SFDO_DRNN.run.migration_cost;
import static Proposed_SFDO_DRNN.run.energy;

public class fitness_CSO {
    public static ArrayList<Double> func(ArrayList<ArrayList<Integer>> migration) {
        ArrayList<Double> fit = new ArrayList();
        double m = (double)Code.Run.PM, n = (double)Code.Run.VM, c = 1.0, Wmax = 1.0;
        double alpha = Math.random(), beta = Math.random(), gamma = Math.random(), delta = Math.random();
        for( int i = 0; i < migration.size(); i++) {
            double M =0.0;      //no. of migrations of VM
            for( int j = 0; j < VM_Migration.size(); j++) {
                if(VM_Migration.get(j) != migration.get(i).get(j))
                    M++;
            }
            double T = System.nanoTime() - st;
            double power_consumed = (p * Wmax) + ((1-p) * Wmax * resource_utilization_C);
            resource_availability = 1 - resource_utilization;
            migration_cost = (1.0 / m) * (M / (c*n));
            energy = (1 / T) * power_consumed;
            double f = (alpha * resource_availability) + (beta * (1-load)) + (gamma * (1-migration_cost)) + (delta * (1-energy));
            fit.add(f);
        }
        return fit;
    }
}
