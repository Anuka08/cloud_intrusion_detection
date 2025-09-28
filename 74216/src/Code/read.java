package Code;

import java.util.ArrayList;
import static Code.Run.Data;
import static Code.Run.target;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class read {
    
    public static void dataset() throws IOException {
        //preprocess_dataset(); // preprocessing the dataset
        
        Data = new ArrayList<>(); target = new ArrayList(); 
        ArrayList<ArrayList<String>> input_data = read_dataset("Processed\\BoT-IoT.csv");
        
        for (int i = 0; i < input_data.size(); i++) {
            ArrayList<Double> tem = new ArrayList();
            for (int j = 0; j < input_data.get(i).size(); j++) {
                if(j==32)   // target atribute (1 - attack, 0 - not attack)
                    target.add(Double.parseDouble(input_data.get(i).get(j)));
                else
                    tem.add(Double.parseDouble(input_data.get(i).get(j)));
            }
            Data.add(tem);
        }
    }
   
    public static void preprocess_dataset() throws IOException {
        
        ArrayList<ArrayList<Double>> Data = new ArrayList<>();
        ArrayList<ArrayList<String>> input_data = read_dataset("dataset\\BoT-IoT.csv");
        input_data = transpose_1(input_data);   // transpose the data to select the attributes
        
        // string data attributes in dataset
        ArrayList<Integer> string_data = new ArrayList();   
        string_data.add(2); string_data.add(3); string_data.add(4);
        string_data.add(5); string_data.add(6); string_data.add(7);
        string_data.add(10); string_data.add(33); string_data.add(34);
        
        // Empty attributes in dataset
        ArrayList<Integer> missed_data = new ArrayList();   
        missed_data.add(16); missed_data.add(17); missed_data.add(21);
        missed_data.add(22); missed_data.add(23); missed_data.add(24);
        
        
        for (int i = 0; i < input_data.size(); i++) {   // each column attribute(since transposed)
            System.out.println(">> "+i);
            if(string_data.contains(i)) Data.add(string_transfer(input_data.get(i)));   // convert string to data by unique value
            else if(missed_data.contains(i)) Data.add(missed_fill(input_data.get(i)));  // fill missed data with 0
            else Data.add(get_data(input_data.get(i))); // else convert to double
        }
        Data = transpose_2(Data);   // re-transposed to get original format
        
        Code.write.write_double_csv(Data, "Processed\\BoT-IoT.csv");
    }
    
    // string to double conversion
    private static ArrayList<Double> get_data(ArrayList<String> data) {
        ArrayList<Double> converted_data = new ArrayList();
        for (int i = 0; i < data.size(); i++) {
            if(data.get(i).length()<1) converted_data.add(0.0);
            converted_data.add(Double.parseDouble(data.get(i)));
        }
        return converted_data;
    }

    // convert string words to data(by its unique)
    private static ArrayList<Double> string_transfer(ArrayList<String> data) {
        
        ArrayList<Double> converted_data = new ArrayList();
        ArrayList<String> uni_data = unique(data);
        for (int i = 0; i < data.size(); i++) {
            converted_data.add((double)uni_data.indexOf(data.get(i)));
        }
        return converted_data;
    }

    // get unique datas in array
    private static ArrayList<String> unique(ArrayList<String> data) {
        ArrayList<String> unique = new ArrayList();
        for (int i = 0; i < data.size(); i++) {
            if(!unique.contains(data.get(i))) unique.add(data.get(i));  // add unique datas
        }
        return unique;
    }

    // fill arraylist with 0
    private static ArrayList<Double> missed_fill(ArrayList<String> data) {
        ArrayList<Double> converted_data = new ArrayList();
        for (int i = 0; i < data.size(); i++) converted_data.add(0.0);  // 0 added to missed data
        return converted_data;
    }
    
    // Reading
    private static ArrayList<ArrayList<String>> read_dataset(String Path) throws FileNotFoundException, IOException {
        ArrayList<ArrayList<String>> data = new ArrayList<>();  
        BufferedReader br = new BufferedReader(new FileReader(Path));
        String s = "";
        while ((s = br.readLine()) != null) {   //read each and every line from the data file
            String a[] = s.split(",");          // since csv data, split by ,
            ArrayList<String> tem = new ArrayList();
            for (int i = 0; i < a.length; i++) {   
                if(a[i].length()<1) tem.add("0");  // replace missing values by 0
                else tem.add(a[i]);         // add data to arraylist
            }
            data.add(tem);      // add each row to data
        }
        return data;
    }
    
    // transpose data
    private static ArrayList<ArrayList<String>> transpose_1(ArrayList<ArrayList<String>> input_data) {
        ArrayList<ArrayList<String>> trans_data = new ArrayList<>();
        for (int i = 0; i < input_data.get(0).size(); i++) {
            ArrayList<String> tem = new ArrayList();
            for (int j = 0; j < input_data.size(); j++) {
                tem.add(input_data.get(j).get(i));
            }
            trans_data.add(tem);
        }
        return trans_data;
    }
    
    public static ArrayList<ArrayList<Double>> transpose_2(ArrayList<ArrayList<Double>> input_data) {
        ArrayList<ArrayList<Double>> trans_data = new ArrayList<>();
        for (int i = 0; i < input_data.get(0).size(); i++) {
            ArrayList<Double> tem = new ArrayList();
            for (int j = 0; j < input_data.size(); j++) {
                tem.add(input_data.get(j).get(i));
            }
            trans_data.add(tem);
        }
        return trans_data;
    }
}
