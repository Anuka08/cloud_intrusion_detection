package SVM;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class Demo {

    public static void testLinearKernel(ArrayList<ArrayList<Double>> data, ArrayList<Double> test_cls) throws IOException, ClassNotFoundException {
        String trainFileName = "training";
        String testFileName = "testing";
        String outputFileName = "op.txt";
        Random r = new Random();
        //Read training file
        Instance[] trainingInstances = DataFileReader.readDataFile(trainFileName, data, test_cls);

        //Register kernel function
        KernelManager.setCustomKernel(new LinearKernel());

        //Setup parameters
        svm_parameter param = new svm_parameter();

        //Read test file
        Instance[] testingInstances = DataFileReader.readDataFile(testFileName, data, test_cls);
    }

    private static void writeOutputs(String outputFileName, double[] predictions) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFileName));
        for (double p : predictions) {
            writer.write(String.format("%.0f\n", p));
        }
        writer.close();
    }

    private static void showUsage() {
    }

    private static boolean checkArgument(String[] args) {
        return args.length == 3;
    }

    public static void classify(ArrayList<ArrayList<Double>> data, ArrayList<Double> clas) throws IOException, ClassNotFoundException {
        
        ArrayList<Double> unique_clas = new ArrayList();
        for (int i = 0; i < clas.size(); i++) {
            if(!unique_clas.contains(clas.get(i)))
                unique_clas.add(clas.get(i));
        }
        
        int trp = (int) ((data.size()*Code.Run.tp)/100);   // training % of data
        ArrayList<ArrayList<Double>> test_data = new ArrayList<>();
        ArrayList<Double> testing_clas = new ArrayList();
        for (int i = 0; i < data.size(); i++) {
            if (i > trp) {
                test_data.add(data.get(i));
                testing_clas.add(clas.get(i));
            }
        }
        
        testLinearKernel(data, testing_clas);
        showUsage();
        ArrayList<Double> res = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader("op.txt"));
        String s = "";

        while ((s = br.readLine()) != null) {
            res.add(Double.parseDouble(s));
        }

        int tp = 0, fp = 0, tn = 0, fn = 0;
        for (int i1 = 0; i1 < unique_clas.size(); i1++) {
            double c = unique_clas.get(i1);
            for (int i = 0; i < res.size(); i++) {
                if (res.get(i) == c && testing_clas.get(i) == c) {
                    tp++;
                }
                if (res.get(i) != c && testing_clas.get(i) != c) {
                    tn++;
                }
                if (res.get(i) == c && testing_clas.get(i) != c) {
                    fn++;
                }
                if (res.get(i) != c && testing_clas.get(i) == c) {
                    fp++;
                }
            }
        }
        tn = tn/unique_clas.size();
        double acc = (double)(tp+tn)/(double)(tp+tn+fp+fn);
        double dr = (double) tp / (double) (tp + tn + fp + fn);
        double fpr =(double)fp/(double)(fp+tn);
        Code.Run.Accuracy.add(acc);
        Code.Run.Detection_Rate.add(dr);
        Code.Run.FPR.add(fpr);
    }
}
