package Proposed_SFDO_DRNN;

import java.util.ArrayList;
import java.util.Collections;
import static Code.Run.VM;
import static Code.Run.PM;
import static Code.Run.VM_Migration;
import static Proposed_SFDO_DRNN.run.VM_Migration_update;

public class chicken_swarm {
    public static ArrayList<Integer> Final_best = new ArrayList();
    public static void main(String[] args) {
        VM_Migration_update = (ArrayList<Integer>) VM_Migration.clone(); //initial VM migration
        Final_best = new ArrayList();
        ArrayList<ArrayList<Integer>> Solution = new ArrayList<>();
        ArrayList<ArrayList<Integer>> Solution_update = new ArrayList<>();
        int i, j, Max_Generation = 1;
        int N = 10;     //row size
        int D = VM;     //column size
        int G = 2;      //time step between(2, 20)       
        
        //initial solution
        for(i = 0; i < N; i++) {
            ArrayList<Integer> tem = new ArrayList();
            for( j = 0; j < VM; j++) {      
                int a = (int)(Math.random()*PM + 1);    //no. of PM in mig.
                tem.add(a);
            }   
            Solution.add(tem);
        }    
        
        int t = 0;
        ArrayList<Integer> rooster = new ArrayList();  
        ArrayList<Integer> hen = new ArrayList();  
        ArrayList<Integer> chick = new ArrayList();  
        ArrayList<Double> Fit = new ArrayList();
        ArrayList<Double> Fit_tem1 = new ArrayList();
        ArrayList<Double> Fit_tem2 = new ArrayList();
        
        //loop
        while(t < Max_Generation) {
            Solution_update = new ArrayList<>();
            
            //Status only update in G TIME STEPS 
            if(t % G == 0) {    
                rooster = new ArrayList();  
                hen = new ArrayList();  
                chick = new ArrayList(); 
                
                //FITNESS calculation
                Fit = new ArrayList();
                Fit_tem1 = new ArrayList();
                Fit_tem2 = new ArrayList();
                Fit = fitness_CSO.func(Solution); 
                Final_best = Solution.get(Fit.indexOf(Collections.max(Fit)));
                VM_Migration_update = Final_best;
                Fit_tem1 = (ArrayList<Double>) Fit.clone();      //for best
                Fit_tem2 = (ArrayList<Double>) Fit.clone();      //for worst
                
                //DIVIDE THE SWARM into groups(Rooster(best), Hen, Chick(worst)) -- based on fitness_CSO(best = min) 
                int m = N / 3;      
                for(i = 0; i < m; i++) {
                //rooster -- best m
                    int r = Fit_tem1.indexOf(Collections.max(Fit_tem1));
                    rooster.add(r); 
                    Fit_tem1.set(r, 0.0);
                //chick -- worst m
                    int c = Fit_tem2.indexOf(Collections.min(Fit_tem2));
                    chick.add(c); 
                    Fit_tem2.set(c, 100000.0);
                }   //end for
                //hen -- others
                for(i = 0; i < N; i++) {
                    if(!rooster.contains(i) && !chick.contains(i)) {
                        hen.add(i);
                    }
                }
            }   //end if

            Solution_update = update_solution(N, D, Fit, rooster, hen, chick, Solution);
            Solution = better_solution(Solution, Solution_update);
            t++;
        }   //end while
        Fit = fitness_CSO.func(Solution);
    }   //end main

    public static ArrayList<ArrayList<Integer>> update_solution(int N, int D, ArrayList<Double> Fit, ArrayList<Integer> rooster, ArrayList<Integer> hen, ArrayList<Integer> chick, ArrayList<ArrayList<Integer>> Solution) {
        //SOLUTION UPDATE
        ArrayList<ArrayList<Integer>> updated_soln = new ArrayList<>();
        for(int i = 0; i < N; i++) {   
            ArrayList<Integer> temp = new ArrayList();
            for(int j = 0; j < D; j++) {
                int epsilon = 1;   //ep.. => small constant to avoid zero

                //rooster update
                if(rooster.contains(i)) {
                    temp = update_rooster(i, N, epsilon, Fit, Solution);
                }

                //hen update
                if(hen.contains(i)) {
                    temp = update_hen(i, N, epsilon, Fit, Solution);
                }

                //chick update
                if(chick.contains(i)) {
                    temp = update_chick(i, N, Solution);
                }
            }   
            updated_soln.add(temp);
        }   //end solution update for
        return updated_soln;
    }
    
    private static ArrayList<Integer> update_rooster(int i, int N, int epsilon, ArrayList<Double> Fit, ArrayList<ArrayList<Integer>> Solution) {
        ArrayList<Integer> tem = new ArrayList();
        //k value belongs to i, but != i
        int k;  //(here k = i+1)
            if(i < N-1) 
                k = (i + 1);
            else
                k = 0;

        //sigma_sq => standard deviation
        int sigma_sq;
            if(Fit.get(i) <= Fit.get(k))
                sigma_sq = 1;
            else
                sigma_sq = (int) Math.exp((Fit.get(k) - Fit.get(i)) / (Math.abs(Fit.get(i)) + epsilon));
            
        //update
        for(int j = 0; j < Solution.get(i).size(); j++) {
            int r;
                r = (int) (Solution.get(i).get(j) * (1 + Math.random()*sigma_sq)); 
            tem.add(Math.abs(r));
        }
        return tem;
    }   //end update_rooster
    
    private static ArrayList<Integer> update_hen(int i, int N, int epsilon, ArrayList<Double> Fit, ArrayList<ArrayList<Integer>> Solution) {
        ArrayList<Integer> tem = new ArrayList();
        //Rand => random no. b/w 0 & 1, r1 & r2 belongs to i but r1 != r2
        int Rand = 0, r = 1, a = 2;
        int r1, r2; 
            if(i < N-1) {
                r1 = (i + 1);
                if(i == N-2)
                    r2 = 0;
                else
                    r2 = (r1 + 1);
            }
            else {
                r1 = 0;
                r2 = r1 + 1;
            }
        
        //S1 & S2
        int S1, S2;
            S1 = (int) Math.exp((Fit.get(i) - Fit.get(r1)) / (Math.abs(Fit.get(i)) + epsilon));
            S2 = (int) Math.exp(Fit.get(r2) - Fit.get(i));
            
        //A & C
        int A, C;
            A = (2 * a * r) - a;
            C = 2 * r;
        
        //PROPOSED UPDATE: ChicWhale (Integrated WOA in CSO)
        for(int j = 0; j < Solution.get(i).size(); j++) {
            int h = (A / (A - 1 + (S1 * Rand) + (S2 * Rand)))
                    * ((S1 * Rand * Solution.get(r1).get(j)) + (S2 * Rand * Solution.get(r2).get(j))
                    - (((Solution.get(i).get(j) * (1 - (A * C))) / A) * (1 - (S1 * Rand) - (S2 * Rand))));
            tem.add(Math.abs(h));
        }
        return tem;
    }   //end update_hen
    
    private static ArrayList<Integer> update_chick(int i, int N, ArrayList<ArrayList<Integer>> Solution) {
        ArrayList<Integer> tem = new ArrayList();
        //FL => random between (0,2), m belongs to i
        int FL = 1;
        int m;  
            if(i < N-1) 
                m = (i + 1);
            else
                m = 0;

        //update
        for(int j = 0; j < Solution.get(i).size(); j++) {
            int c =  Solution.get(i).get(j) + (FL * (Solution.get(m).get(j) - Solution.get(i).get(j)));
            tem.add(Math.abs(c));
        }
        return tem;
    }   //end update_chick
    
    public static ArrayList<ArrayList<Integer>> better_solution(ArrayList<ArrayList<Integer>> Solution, ArrayList<ArrayList<Integer>> Solution_update) {
        ArrayList<ArrayList<Integer>> best_soln = new ArrayList<>();
        
        ArrayList<Double> soln_fit = new ArrayList(); 
            soln_fit = fitness_CSO.func(Solution);      //Fitness of Solution
        
        ArrayList<Double> soln_up_fit = new ArrayList(); 
            soln_up_fit = fitness_CSO.func(Solution_update);    //Fitness of updated_soln
        
        //for each row solution with max. fitness_CSO is taken as best_solution
        for(int i = 0; i < Solution.size(); i++) {
            if(soln_fit.get(i) > soln_up_fit.get(i))
                best_soln.add(Solution.get(i));
            else
                best_soln.add(Solution_update.get(i));
        }
        return best_soln;   //(each Solution with max fitness_CSO)
    }   //end better_solution
}   //end class
