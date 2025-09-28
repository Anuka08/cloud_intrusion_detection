package Proposed_SFDO_DRNN;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class SailFish_update {
    
    public static ArrayList<Double> acc = new ArrayList();
    public static ArrayList<Double> dr = new ArrayList();
    public static ArrayList<Double> fpr = new ArrayList();
    
    public static void optimization(ArrayList<ArrayList<Double>> feature, ArrayList<Double> clas) throws Exception {
   
        int NP = 2, D = 5, Max = 10, di = 100, epoch = 0;       // population, dimension, max. epoch & epsilon => initial parameters
        double A = 4.0, ep = 0.001;                             // n_variables, A
        
        ArrayList<ArrayList<Double>> SF = generate_soln(NP, D); // initial SailFish
        ArrayList<ArrayList<Double>> S = generate_soln(NP, D);  // initial Sardine
        ArrayList<Double> SF_Fit = SFDO_fitness.func(SF, feature, clas);    // fitness
        ArrayList<Double> S_Fit = SFDO_fitness.func(S, feature, clas);   
        
        int bst = SF_Fit.indexOf(Collections.min(SF_Fit));              // get best (Minimization)
        int s_bst = S_Fit.indexOf(Collections.min(S_Fit));
        
        double Best_Fit = SF_Fit.get(bst), S_best = S_Fit.get(s_bst);   // best fitness(SF), best fitness(S)
        ArrayList<Double> elite_SF = SF.get(bst);                       // elite SailFish (Global best)
        ArrayList<Double> Pp = SF.get(bst);                             // Personal best
        ArrayList<Double> injured_S = S.get(s_bst);                     // injured Sardine 
        
        // loop begins
        System.out.println("\nPlease wait it will take some time(nearly 30 mins)...");
        while (epoch  < Max) {
            SF = update_SF(SF, S, elite_SF, injured_S, Pp);     // update position of SF -- Proposed update
            double AP = A * (1.0-(2.0*(epoch+1)*ep));           // Attack Power (Eq. 10)
            
            if (AP < 0.5) {     // update position of selected sardine (Eq. 9)
                int alpha = (int)(S.size()*AP);         // alpha calculation (Eq. 11)
                int beta = (int)(di*AP);                // beta calculation (Eq. 12)
                int set_1 = random_choice(alpha, S);    // random set of sardine
                int set_2 = random_choice(beta, S);
                
                // update position of selected sardine (Eq. 9)
                for (int j = 0; j < S.get(set_1).size(); j++) {
                    S.get(set_1).set(j, (Math.random() * (elite_SF.get(j)-S.get(set_1).get(j)+AP)));
                    S.get(set_2).set(j, (Math.random() * (elite_SF.get(j)-S.get(set_2).get(j)+AP)));
                }
            }
            
            else {  // update position of all sardine (Eq. 9)
                for (int i = 0; i < S.size(); i++) {
                    for (int j = 0; j < S.get(i).size(); j++) {
                        S.get(i).set(j, (Math.random() * (elite_SF.get(j)-S.get(i).get(j)+AP)));
                    }
                }
            }
            SF_Fit = SFDO_fitness.func(SF, feature, clas);  // fitness
            S_Fit = SFDO_fitness.func(S, feature, clas);  
            
            bst = SF_Fit.indexOf(Collections.min(SF_Fit));  // get best
            s_bst = S_Fit.indexOf(Collections.min(S_Fit));
            Pp = SF.get(bst);               // Personal best
            
            // Update the best SailFish & best sardine
            if(SF_Fit.get(bst) < Best_Fit) {
                Best_Fit = SF_Fit.get(bst);
                elite_SF = SF.get(bst);     // Global best
            }
            if (S_Fit.get(s_bst) < S_best) {
                S_best = S_Fit.get(s_bst);
                injured_S = S.get(s_bst);
            }
            epoch++;
        }
        
        Code.Run.Accuracy.add(avg(acc));
        Code.Run.Detection_Rate.add(avg(dr));
        Code.Run.FPR.add(avg(fpr));
    }

    // Generating random solution
    private static ArrayList<ArrayList<Double>> generate_soln(int NP, int D) {
        ArrayList<ArrayList<Double>> data = new ArrayList<>();
        for (int i = 0; i < NP; i++) {
            ArrayList<Double> tem = new ArrayList();
            for (int j = 0; j < D; j++) {
                tem.add(Math.random());
            }
            data.add(tem);
        }
        return data;
    }

    // SailFish update
    private static ArrayList<ArrayList<Double>> update_SF(ArrayList<ArrayList<Double>> SF, ArrayList<ArrayList<Double>> S, ArrayList<Double> elite_SF, ArrayList<Double> injured_S, ArrayList<Double> Pp) {
                
        // for each sailfish calculate lamda_i (Eq. 7)
        ArrayList<ArrayList<Double>> new_SF = new ArrayList<>();
        for (int i = 0; i < SF.size(); i++) {
            double PD = 1.0 - (SF.size()/(SF.size()+S.size()));
            double lamda_i = 2.0*Math.random()*PD - PD;
            double B1p = Math.random(), B2p = Math.random();    // random(0,1)
            
            // Update the position of sailfish (Eq.6)
            ArrayList<Double> tem = new ArrayList();
            for (int j = 0; j < SF.get(i).size(); j++) {
                double G = elite_SF.get(j), P = Pp.get(j), dpj = S.get(i).get(j);  // Global & Personal best
                ///////////////////// Proposed Update: SFDO (Sailfish+Dolphin) ////////////////////////
                double U = ((1.0-B1p-B2p)/(2.0-B1p-B2p)) * (elite_SF.get(j) - lamda_i * (Math.random()*
                        ((elite_SF.get(j)+injured_S.get(j))/2.0)-((dpj-(B1p*P)-(B2p*G))/(1.0-B1p-B2p))));
                tem.add(U);
            }
            new_SF.add(tem);
        }
        return new_SF;
    }

    private static int random_choice(int A, ArrayList<ArrayList<Double>> S) {
        int s;
        Random r = new Random();
        if (A>=0 && A<(S.size()-1)) s = A;
        else s = r.nextInt(S.size());
        return s;
    }
    
    private static double avg(ArrayList<Double> data) {
        double summ = 0;
        for (int i = 0; i < data.size(); i++) {
            summ += data.get(i);
        }
        return summ/data.size();
    }
}
