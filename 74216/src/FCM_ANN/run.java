package FCM_ANN;

import java.util.ArrayList;
import java.util.Collections;
import static Code.Run.tp;

public class run {

    public static void callmain(ArrayList<ArrayList<Double>> data, ArrayList<Double> clas) {
        Percepton.initialize_data(data);
        ArrayList<Double> weights = Percepton.INITIAL_WEIGHTS;
        ArrayList<Double> predicted=new ArrayList<Double>();
        run driver = new run();
        Percepton percepton = new Percepton();
        int epochNumber = 0;
        double error;
        int p = (data.size() * (tp - 10))/100;
        for (int x = 0; x < data.size(); x++) {
            double weightedSum = percepton.calculateWeightedSum(data.get(x), weights);
            double result = percepton.applyActivationFunction(weightedSum, clas.get(x), p, x, data, clas);
            predicted.add(result);
        }
        ArrayList<Double> unique_clas = new ArrayList();
        for (int i  = 0; i < clas.size(); i++) {
            if (!unique_clas.contains(clas.get(i)))
                unique_clas.add(clas.get(i));
        }
        
        int tp=0,tn=0,fp=0,fn=0;
        for(int i1 = 0; i1 < unique_clas.size(); i1++)
        {
            double c = unique_clas.get(i1);
            for(int i=0;i<clas.size();i++)
            {
                if(clas.get(i) == c  && predicted.get(i) == c){
                tp++;}
                if(clas.get(i) != c  && predicted.get(i) != c){
                tn++;}
                if(clas.get(i) == c  && predicted.get(i) != c){
                fn++;}
                if(clas.get(i) != c  && predicted.get(i) == c){
                fp++;}
            }
        } tn = tn/unique_clas.size();
        double acc = (double) (tp + tn) / (double) (tp + tn + fp + fn);
        double dr = (double) tp / (double) (clas.size());
        double fpr=(double)fp/(double)(fp+tn);

        Code.Run.Accuracy.add(acc); 
        Collections.sort(Code.Run.Accuracy);
        Code.Run.Detection_Rate.add(dr); 
        Collections.sort(Code.Run.Detection_Rate);
        Code.Run.FPR.add(fpr); 
        Collections.sort(Code.Run.FPR, Collections.reverseOrder());
    }
}