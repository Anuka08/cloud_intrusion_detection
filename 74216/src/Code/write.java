package Code;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class write {
    
    //write 2D double data to csv file
    public static void write_double_csv(ArrayList<ArrayList<Double>> data, String path) throws IOException {
        FileWriter fw = new FileWriter(path);
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < data.get(i).size(); j++) {
                fw.write(data.get(i).get(j) + ",");
            }
            fw.write(System.lineSeparator());
            fw.flush();
        }
    }
}
