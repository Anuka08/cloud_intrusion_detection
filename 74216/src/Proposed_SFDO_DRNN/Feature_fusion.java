package Proposed_SFDO_DRNN;

import static Code.Run.Data;
import static Code.Run.target;
import static Code.Run.Feature;
import java.io.IOException;
import java.util.ArrayList;

public class Feature_fusion {
        
    public static void process() throws IOException, Exception {
        
        int n_cluster = Code.Run.Group_size;
        System.out.println("\nReading data..");
        Code.read.dataset();
        
        System.out.println("\nFeature Fusion..");
        System.out.println("\t>> Feature grouping by FCM..Please wait...");
        ArrayList<Integer> Cluster = Code.FCM.group(Code.Run.Data, n_cluster);
        
        System.out.println("\t>> Feature fusion of every group..");
        Feature = new ArrayList<>();
        for (int i = 0; i < n_cluster; i++) {
            ArrayList<Double> tem = new ArrayList();
            double alpha = Math.random();   // constant
            for (int j = 0; j < Data.size(); j++) {
                tem.add(fuse_feature(Data.get(j), Cluster, i+1, alpha));    // fusing features in same group
            }
            Feature.add(tem);
        }
        Feature = Code.read.transpose_2(Feature);   // transpose to get data in original fused format
        Code.write.write_double_csv(Feature, "Processed\\Fused_Feature.csv");
        
        System.out.println("\nIntrusion detection by SFDO based DRNN..");
        SailFish_update.optimization(Feature, target);
    }
    
    // Feature fusion
    private static double fuse_feature(ArrayList<Double> F, ArrayList<Integer> Clustered, int cluster_n, double ai) {
        
        double fused_data =  0;
        for (int i = 0; i < Clustered.size(); i++) {
            if (Clustered.get(i)==cluster_n) {          // if feature attribute in current cluster
                fused_data += ((1.0/ai) * F.get(i));    // formula for feature fusion 
            }
        }
        return fused_data;
    }
}
