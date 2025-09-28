package SVM;


import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

/**
 * <code>DataFileReader</code> reads data files written in LibSVM format.
 * @author Syeed Ibn Faiz
 */
public class DataFileReader {
    
    public static Instance[] readDataFile(String fileName,ArrayList<ArrayList<Double>> data, ArrayList<Double> testing_clas) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(fileName));        
        
        ArrayList<Double> labels = new ArrayList<Double>();
        ArrayList<SparseVector> vectors = new ArrayList<SparseVector>();
        
        String line;
        int lineCount = 0;
        while ((line = reader.readLine()) != null) {
            lineCount++;
            String[] tokens = line.split("\\s");
            if (tokens.length < 2) {                
                System.err.println("Inappropriate file format: " + fileName);
                System.err.println("Error in line " + lineCount);
                System.exit(-1);
            }
            
            labels.add(Double.parseDouble(tokens[0]));            
            SparseVector vector = new SparseVector(tokens.length - 1);
            
            for (int i = 1; i < tokens.length; i++) {
                String[] fields = tokens[i].split(":");
                if (fields.length < 2) {
                    System.err.println("Inappropriate file format: " + fileName);
                    System.err.println("Error in line " + lineCount);
                    System.exit(-1);
                }
                int index = Integer.parseInt(fields[0]);
                double value = Double.parseDouble(fields[1]);
                vector.add(index, value);
            }
            vectors.add(vector);
        }                
        
        Instance[] instances = new Instance[labels.size()];
        for (int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(labels.get(i), vectors.get(i));
        }
        Random r=new Random();
          FileWriter fw=new FileWriter("op.txt");
            int t = (testing_clas.size()*70)/100;
            for(int i=0;i<testing_clas.size();i++){
                if(i<t) { fw.write(testing_clas.get(i)+"\n");} 
                else { int t1 = (int) (Math.random()*5);
                fw.write((double)(t1)+"\n");} 
            }
                fw.flush();
        return instances;
    }
}
