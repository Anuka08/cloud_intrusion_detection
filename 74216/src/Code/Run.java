package Code;

import java.io.IOException;
import java.util.ArrayList;

public class Run {
    
    public static double th = 0.04;              //threshold
    public static int PM = 10, VM = 50;         //initialize Physical & virtual machine
    public static int task = Code.GUI.task, max_itr = Code.GUI.iteration;     //input from GUI
    public static ArrayList<Integer> VM_Migration = new ArrayList();
    public static ArrayList<Double> P = new ArrayList();    //Processing entities
    public static ArrayList<Double> C = new ArrayList();    //CPU
    public static ArrayList<Double> B = new ArrayList();    //Bandwidth
    public static ArrayList<Double> M = new ArrayList();    //Memory
    public static ArrayList<Double> I = new ArrayList();    //MIPS
    public static int Group_size = Code.GUI.gp, tp = Code.GUI.tr_per;  // Group size, training percentage
    public static ArrayList<Double> Accuracy = new ArrayList();         // Accuracy
    public static ArrayList<Double> Detection_Rate = new ArrayList();   // Detection rate
    public static ArrayList<Double> FPR = new ArrayList();              // FPR
    public static ArrayList<ArrayList<Double>> Data = new ArrayList<>();    // Data
    public static ArrayList<Double> target = new ArrayList();               // target
    public static ArrayList<ArrayList<Double>> Feature = new ArrayList<>(); // Fused Feature
    
    public static void callmain() throws IOException, Exception {
        
        Accuracy = new ArrayList(); Detection_Rate = new ArrayList(); FPR = new ArrayList(); 
        Data = new ArrayList<>(); target = new ArrayList(); 
        
        //initializing the VM_parameters
        P = new ArrayList(); C = new ArrayList(); B = new ArrayList(); M = new ArrayList(); I = new ArrayList();
        System.out.println("\nInitializing VM..");
        initial_VM_migration();
        System.out.println("\nGenerating VM parameters..");
        generate_VM_parameters();
        System.out.println("\nAssign Task..");
        System.out.println("\nCalculating Load & optimizing...");
        Proposed_SFDO_DRNN.run.callmain();
        SVM.Demo.classify(Data, target);
        ANFIS.Process_fuzz.classify(Data, target);
        FCM_ANN.run.callmain(Feature, target);
        System.out.println("\nDone.!");
    }

    public static void initial_VM_migration() {
        int i, j;
        VM_Migration = new ArrayList(); //VM_Mig size 1xVM (1x50)
        int n = 1, m = VM / PM;        //VM in each PM (if VM = 50 & PM = 10, then 5 VM in each PM)
        for( i = 0; i < PM; i++) {      //no. of PM in mig.
            if( n < PM ) {              //if m = 5, then 1st 5 VM in PM 1, 2nd 5 in PM 2, so on..
                int it = 0;
                while( it < m) {
                    VM_Migration.add(n);        //VM in which PM is added to VM_migration
                    it++;
                }
                n++;
            }
            else {                      //last 5 + remaining VM in last PM (if VM = 52 then 7 VM in last PM)
                int it = VM_Migration.size() - 1;
                while( it < VM - 1) {
                    VM_Migration.add(n);
                    it++;
                }
            }
        }
    }

    public static void generate_VM_parameters() {
        //size of paramters = no.of VM, Values range between 1 to 10 (random)
        for( int i = 0; i < VM; i++) {
            P.add(Math.random()*10+1);
            C.add(Math.random()*10+1);
            B.add(Math.random()*10+1);
            M.add(Math.random()*10+1);
            I.add(Math.random()*10+1);
        }
    }
}
