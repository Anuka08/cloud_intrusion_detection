package Code;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class FCM {
    
    public static ArrayList<ArrayList<Double>> U = new ArrayList<>();
    public static ArrayList<Double> obj_fcn = new ArrayList();        //Array for objective function
    
    public static ArrayList<Integer> group(ArrayList<ArrayList<Double>> data, int cluster_n) throws IOException {
        data = Code.read.transpose_2(data);
        ArrayList<ArrayList<Double>> U_trans = new ArrayList<>();
        ArrayList<Integer> Cluster = new ArrayList();
        
        int data_n = data.size();   //data size
        int i, j, expo = 2;         // Exponent for U
        int max_iter = 2;           // Max. iteration
        double min_impro = 1e-5;    // Min. improvement
        
        U = new ArrayList<>();
        U = initfcm(cluster_n, data_n);     // Initial fuzzy partition
               
        int g = 0;
        while ( g < max_iter) {             // Main loop
            U = stepfcm(data, U, cluster_n, expo);  //output : U, center, obj_fcn(i)
            
            if (g > 1)
            {
		if (Math.abs(obj_fcn.get(g) - obj_fcn.get(g-1)) < min_impro)    //termination condition
                {
                    g = max_iter;
                }
            }
            g++;
        }

        U_trans = transpose(U);
                                    
        for( i = 0; i < U_trans.size(); i++)
        {
            ArrayList<Double> tem = new ArrayList();
            tem = U_trans.get(i);
            int m = tem.indexOf(Collections.max(tem)); 
            Cluster.add(m+1);
        }
        
        if (unique(Cluster).size() != cluster_n) {  // if no data in any cluster
            for (int i1 = 0; i1 < cluster_n; i1++) {
                Cluster.set(i1, i1+1);
            }
        }
        
        return Cluster;
    }
    
    
    public static ArrayList<ArrayList<Double>> readinput() throws FileNotFoundException, IOException
    {
        int i, j;
        BufferedReader br=new BufferedReader(new FileReader("feature.csv"));    //read input from excel
        ArrayList<ArrayList<Double>> input=new ArrayList<>();
        String s="";                                                            
        while((s=br.readLine())!=null)              //read input in s
        {
            String a[]=s.split(",");                    //comma to read input 1 by 1 
            ArrayList<Double> tem=new ArrayList();      //Single Diamensional array

            for(i=0;i<a.length;i++)                     //a.length =>size of array a,a.size() for arraylist
            {
                tem.add(Double.parseDouble(a[i]));      //add row of data to tem 1 by 1

            }
            input.add(tem);                             //add full matrix data to 2D arraylist "input"
            }
        return input;
    }
    

    public static ArrayList<ArrayList<Double>> initfcm(int cluster_n, int data_n)
    {
        int i, j;
        ArrayList<ArrayList<Double>> U_part = new ArrayList<>();
        ArrayList<ArrayList<Double>> U_Transpose = new ArrayList<>();
        ArrayList<Double> col_sum = new ArrayList();
        ArrayList<ArrayList<Double>> col_sum_change = new ArrayList<>();
        for( i = 0; i < cluster_n; i++)
        {
            ArrayList<Double> tem = new ArrayList();
            for( j =0 ; j < data_n; j++)
            {
                double a =Math.random();
                tem.add(a);
            }
            U_part.add(tem);
        }
        
        U_Transpose = transpose(U_part);
        for( i = 0; i < U_Transpose.size(); i++)
        {
            double sum = 0;
            for( j = 0; j < U_Transpose.get(i).size(); j++)
            {
                sum += U_Transpose.get(i).get(j);
            }
            col_sum.add(sum);
        }
        
        for( i = 0; i < cluster_n; i++)
        {
           col_sum_change.add(col_sum);                 //make col_sum size = U_part size = cluster size
        }
       
       
        // elementwise division
        for( i = 0; i < U_part.size(); i++)
        {
            ArrayList<Double> tem = new ArrayList();
            for( j = 0; j < U_part.get(i).size(); j++)
            {
                double a = U_part.get(i).get(j) / col_sum_change.get(i).get(j);
                tem.add(a);
            }
            U.add(tem);
        }
        return U;
    }
    
    public static ArrayList<ArrayList<Double>> transpose(ArrayList<ArrayList<Double>> Input)
    {
        int i, j;
        ArrayList<ArrayList<Double>> Transpose = new ArrayList<ArrayList<Double>>();
        final int N = Input.get(0).size();              //getting column size
        for ( i = 0; i < N; i++ )                   //assign column size as row size
        {
            ArrayList<Double> col = new ArrayList<Double>();
            for (List<Double> row : Input)              //row <= contains each row & loop until size of Input
            {
                col.add(row.get(i));                //get 1(ith) element from rows each time (for getting column values)
            }
            Transpose.add(col);
        }
        return Transpose;
    }

    public static ArrayList<ArrayList<Double>> stepfcm(ArrayList<ArrayList<Double>> data, ArrayList<ArrayList<Double>> U, int cluster_n, int expo)
    {
        int i, j, k;
        ArrayList<Double> tmp_sum = new ArrayList();
        ArrayList<ArrayList<Double>> mf = new ArrayList<>();
        ArrayList<ArrayList<Double>> matmul_mf_data = new ArrayList<>();
        ArrayList<ArrayList<Double>> matmul_row_mf_sum_ones = new ArrayList<>();
        ArrayList<ArrayList<Double>> row_mf_sum = new ArrayList<>();
        ArrayList<ArrayList<Double>> ones = new ArrayList<>();
        ArrayList<ArrayList<Double>> center = new ArrayList<>();
        ArrayList<ArrayList<Double>> dist = new ArrayList<>();
        ArrayList<ArrayList<Double>> dist_mul_mf = new ArrayList<>();
        ArrayList<ArrayList<Double>> tmp_sum_change = new ArrayList<>();
        ArrayList<ArrayList<Double>> U_new = new ArrayList<>();
        for( i = 0; i < U.size(); i++)
        {
            ArrayList<Double> tem = new ArrayList();
            for( j = 0; j < U.get(i).size(); j++)
            {
                double a = Math.pow(U.get(i).get(j), expo);
                tem.add(a);
            }
            mf.add(tem);
        }
        

        //mf * data
        for (i = 0; i < mf.size(); i++)
        {
            ArrayList<Double> tem=new ArrayList();
            for (j = 0; j < data.get(0).size(); j++)
            {  
               double a=0;
               for (k = 0; k < data.size(); k++)
                {
                  a = a + mf.get(i).get(k) * data.get(k).get(j);        // matrix mul of mf and data 
                }
                tem.add(a);
            }
            matmul_mf_data.add(tem);
        }
        
                                    
        for( i = 0; i < mf.size(); i++)
        {
            double sum = 0;
            ArrayList<Double> tem = new ArrayList();
            for( j = 0; j < mf.get(i).size(); j++)
            {
                sum += mf.get(i).get(j);                    //adding attributes of the row to make single column
            }
            tem.add(sum);
            row_mf_sum.add(tem);
        }
                                    
        for( i = 0; i < 1; i++)
        {
            ArrayList<Double> tem = new ArrayList();
            for( j = 0; j < data.get(0).size(); j++)
            {
                double a = 1.0;
                tem.add(a);
            }
            ones.add(tem);
        }

        //row_mf_sum * ones
        for (i = 0; i < row_mf_sum.size(); i++)
        {
            ArrayList<Double> tem=new ArrayList();
            for (j = 0; j < ones.get(0).size(); j++)
            {  
               double a=0;
               for (k = 0; k < ones.size(); k++)
                {
                  a = a + row_mf_sum.get(i).get(k) * ones.get(k).get(j);        // matrix mul of row_mf_sum and ones
                }
                tem.add(a);
            }
            matmul_row_mf_sum_ones.add(tem);
        }
        
                                        
        //elementwise division
        for( i = 0; i < matmul_mf_data.size(); i++)
        {
            ArrayList<Double> tem = new ArrayList();
            for( j = 0; j < matmul_mf_data.get(i).size(); j++)
            {
                double a = matmul_mf_data.get(i).get(j) / matmul_row_mf_sum_ones.get(i).get(j);
                tem.add(a);
            }
            center.add(tem);
        }
    
        dist = distfcm(center, data);
    
        double a = 0;
        for( i = 0; i < dist.size(); i++)
        {
            ArrayList<Double> tem = new ArrayList();
            for( j = 0; j < dist.get(i).size(); j++)
            {
                a += Math.pow( dist.get(i).get(j), 2 ) * mf.get(i).get(j);
            }
        }
        obj_fcn.add(a);
        
        ArrayList<ArrayList<Double>> tmp = new ArrayList<>();
        ArrayList<ArrayList<Double>> tmp_trans = new ArrayList<>();
        //tmp = dist.^(-2/(expo-1));
        for( i = 0; i < dist.size(); i++)
        {
            ArrayList<Double> tem = new ArrayList();
            for( j = 0; j < dist.get(i).size(); j++)
            {
                double t = Math.pow( dist.get(i).get(j), (-2/(expo-1)) );
                tem.add(t);
            }
            tmp.add(tem);
        }
                                    
        tmp_trans = transpose(tmp);
        for( i = 0; i < tmp_trans.size(); i++)
        {
            double sum = 0;
            for( j = 0; j < tmp_trans.get(i).size(); j++)
            {
                sum += tmp_trans.get(i).get(j);
            }
            tmp_sum.add(sum);
        }
                               
        for( i = 0; i < cluster_n; i++)
        {
           tmp_sum_change.add(tmp_sum);                 //make tmp_sum size= cluster size
        }
        
        //tmp / tmp_sum_change -- (elementwise)
        for( i = 0; i < tmp.size(); i++)
        {
            ArrayList<Double> tem = new ArrayList();
            for( j = 0; j < tmp.get(i).size(); j++)
            {
                double u = tmp.get(i).get(j) / tmp_sum_change.get(i).get(j);
                tem.add(u * Math.random());
            }
            U_new.add(tem);
        }
        return U_new;
        
    }

//    DISTFCM Distance measure in fuzzy c-mean clustering.
//        OUT = DISTFCM(CENTER, DATA) calculates the Euclidean distance between each row in CENTER and each row in DATA, 
//        and returns a distance matrix OUT of size M by N, where M and N are row dimensions of CENTER and DATA, respectively, 
//        and OUT(I, J) is the distance between CENTER(I,:) and DATA(J,:)
    public static ArrayList<ArrayList<Double>> distfcm(ArrayList<ArrayList<Double>> center, ArrayList<ArrayList<Double>> data)
    {
        ArrayList<ArrayList<Double>> out = new ArrayList<>();   // out of size center size x data size
        ArrayList<ArrayList<Double>> ones = new ArrayList<>();
        ArrayList<ArrayList<Double>> matmul_ones_center = new ArrayList<>();
        ArrayList<ArrayList<Double>> data_minus_ones = new ArrayList<>();
        
        int i, j, k, l;

        //fill the output matrix
        for( i = 0; i < data.size(); i++)
        {
            ArrayList<Double> tem = new ArrayList();
            for( j = 0; j < 1; j++)
            {
                double a = 1.0;
                tem.add(a);
            }
            ones.add(tem);                       // ones of size data size x 1
        }
                                            
        
        for( k = 0; k < center.size(); k++ )
        {
            //matrix mul of ones and center or copy each row of center to data size (to make center size = data size)
            matmul_ones_center = new ArrayList<>();
            for (i = 0; i < data.size(); i++)
            {
                matmul_ones_center.add(center.get(k));
            }
                                        
            //(data - matmul_ones_center) ^ 2 -- (elementwise)
            data_minus_ones = new ArrayList<>();
            for (i = 0; i < data.size(); i++)
            {
                ArrayList<Double> tem=new ArrayList();
                for (j = 0; j < data.get(i).size(); j++)
                {
                    double a = Math.pow(data.get(i).get(j) - matmul_ones_center.get(i).get(j), 2);
                    tem.add(a);
                }
                data_minus_ones.add(tem);
            }
                                        
            //sum elements of row, then take square root
            ArrayList<Double> tem = new ArrayList();
            for( i = 0; i < data_minus_ones.size(); i++)
            {
                double sum = 0;
                for( j = 0; j < data_minus_ones.get(i).size(); j++)
                {
                    sum += data_minus_ones.get(i).get(j);                   //to make single column
                }
                tem.add( Math.sqrt(sum) );      //take sqrt of each data
            }
            out.add(tem);
        }
            
        return out;
    }
    
    // get unique datas in array
    private static ArrayList<Integer> unique(ArrayList<Integer> data) {
        ArrayList<Integer> unique = new ArrayList();
        for (int i = 0; i < data.size(); i++) {
            if(!unique.contains(data.get(i))) unique.add(data.get(i));  // add unique datas
        }
        return unique;
    }
}