package Proposed_SFDO_DRNN;

import static Code.Run.task;
import static Code.Run.VM;
import static Code.Run.P;
import static Code.Run.C;
import static Code.Run.B;
import static Code.Run.Data;
import static Code.Run.M;
import static Code.Run.I;
import static Code.Run.VM_Migration;
import static Code.Run.th;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

public class run {
    public static double resource_utilization, resource_utilization_C, load, resource_availability, migration_cost = 1.0, energy, st, p;
    public static ArrayList<Integer> task_time = new ArrayList();
    public static ArrayList<Integer> task_assign = new ArrayList();
    public static ArrayList<Integer> VM_Migration_update = new ArrayList();
    public static void callmain() throws IOException, Exception {
        
        int g = 0;
        st = System.nanoTime();     //start time 
        task_assign = new ArrayList();
        VM_Migration_update = (ArrayList<Integer>) VM_Migration.clone();
        while(g < Code.Run.max_itr) {
            assign_task_time(g);
            VM_task_assign(g);
            p = System.nanoTime() - st;  //time limit
            load_calculation(st);
            if(load > th) {
                chicken_swarm.main(null);
            }
            
            for( int i = 0; i < task_assign.size(); i++) {
                if(task_assign.get(i) > 0)
                task_assign.set(i, (task_assign.get(i) - 1));  
            }
            g++;
            compute_parameter();
        }
        
        // Proposed II - begins
        Feature_fusion.process();
    }
    
    public static void assign_task_time(int g) {
        //assign time to each task (randomly from 1 to max_itr) 
        if( g == 0) {
            for( int i = 0; i < task; i++) {
                task_time.add((int)(Math.random()*Code.Run.max_itr+1));      
            }
        }
    }

    public static void VM_task_assign(int g) {
        /*assign task to VM based on round-robin fashion(one by one), 
        remaining waiting task can be assigned in free VM(which completed its before task)*/
        if(g == 0) {
            //System.out.println("task_time : "+task_time);
            for(int i = 0; i < VM; i++) {
                task_assign.add(0);     //initialize task_assign to the size of VM
            }
        }
        for(int i = 0; i < task_assign.size(); i++) {
            if(task_time.size() > 0) {          //check task availability
                if(task_assign.get(i) == 0) {   //check VM availability(free VM)
                    task_assign.set(i, task_time.get(0));   //set the task to VM
                    task_time.remove(0);        //remove added task from task_time arraylist(since it is added in VM)
                }
            }
        }
    }

    public static void load_calculation(double st) {
        resource_utilization = 0.0; load = 0.0;     //initializing
        double N = 300, time, m = Code.Run.PM;   //N => Normalizing factor
        
        for(int i = 0; i < VM; i++) {       //for the no. of VM
            if(task_assign.get(i) > 0) {    //summation of the VM parameter of active VM (having task)
                resource_utilization += ((P.get(i)/Collections.max(P)) + (C.get(i)/Collections.max(C)) + (B.get(i)/Collections.max(B)) + (M.get(i)/Collections.max(M)) + (I.get(i)/Collections.max(I)));
            }
        }
        resource_utilization = resource_utilization / N;
        resource_utilization_C = resource_utilization / m;
        time = (System.nanoTime() - st) / 1000000;      //time from process starts
        load = resource_utilization / time;
    }

    public static void compute_parameter() {
        double m = (double)Code.Run.PM, n = (double)Code.Run.VM, c = Math.random()*1+0.5, Wmax = 0.5;
        double alpha = Math.random(), beta = Math.random(), gamma = Math.random(), delta = Math.random();
        
        double M =0.0;      //no. of migrations of VM
        for( int j = 0; j < VM_Migration.size(); j++) {
            if(VM_Migration.get(j) != VM_Migration_update.get(j))
                M++;
        }
        double T = System.nanoTime() - st;
        double power_consumed = (p * Wmax) + ((1-p) * Wmax * resource_utilization_C);
        resource_availability = 1 - resource_utilization;
        migration_cost = (1.0 / m) * (M / (c*n));
        energy = (1 / T) * power_consumed;
    }
}
