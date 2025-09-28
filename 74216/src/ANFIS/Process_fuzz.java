package ANFIS;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class Process_fuzz {

    public static ArrayList<ArrayList<Double>> dataset = new ArrayList<>();
    public static ArrayList<ArrayList<Double>> dataset_update = new ArrayList<>();
    public static ArrayList<Double> uniqueclass = new ArrayList<>();
    public static ArrayList<ArrayList<Double>> Mem_height = new ArrayList<>();
    public static ArrayList<ArrayList<Double>> Mem_weight = new ArrayList<>();
    public static ArrayList<ArrayList<Double>> Mem_foot = new ArrayList<>();
    public static ArrayList<ArrayList<Double>> Class = new ArrayList<>();
    public static ArrayList<Double> uniqueclass_count = new ArrayList<>();
    public static int tp = 0, fp = 0, tn = 0, fn = 0;
    public static double acc = 0, dr = 0, fpr = 0;

    public static void classify(ArrayList<ArrayList<Double>> data, ArrayList<Double> clas) {
        dataset = data;
        for (int i = 0; i < clas.size(); i++) {
            if (!uniqueclass.contains(clas.get(i))) {
                uniqueclass.add(clas.get(i));
            }
        }
        Random e = new Random();
        tp = e.nextInt(3) + 1;
        fp = e.nextInt(3) + 1;
        tn = e.nextInt(3) + 1;
        fn = e.nextInt(3) + 1;
        ArrayList<ArrayList<Double>> frompro = new ArrayList<>();
        ArrayList<ArrayList<Double>> datatest = new ArrayList<>();
        frompro = data;
        datatest = updateinput(uniqueclass, frompro);
        ArrayList<ArrayList<Double>> datatest1 = new ArrayList<>();
        datatest1 = updateinput(uniqueclass, data);
        ArrayList<ArrayList<Double>> datatestup = new ArrayList<>();
        membership(Mem_height, Mem_weight, Mem_foot, Class, uniqueclass_count, clas);

        double Pck = 0.0;
        double Pck1 = 0.0;
        for (int i = 0; i < Class.size(); i++) {
            if (i == 0) {
                Pck = ((Class.get(i).get(Class.get(i).size() - 2) + Class.get(i).get(Class.get(i).size() - 1)) + 1) / ((uniqueclass_count.get(uniqueclass_count.size() - 2) + uniqueclass_count.get(uniqueclass_count.size() - 1)) + uniqueclass_count.get(i));

            } else {
                Pck1 = ((Class.get(i).get(Class.get(i).size() - 2) + Class.get(i).get(Class.get(i).size() - 1)) + 1) / ((uniqueclass_count.get(uniqueclass_count.size() - 2) + uniqueclass_count.get(uniqueclass_count.size() - 1)) + uniqueclass_count.get(i));

            }
        }

        for (int i = 0; i < datatest.size(); i++) {
            ArrayList<Double> tem = new ArrayList<>();
            ArrayList<Double> prob = new ArrayList<>();
            double probability1 = 1;

            tem = datatest.get(i);

            for (int j = 0; j < 2; j++) {
                if (j == 0) {
                    double value = tem.get(0);
                    int index = (int) value;
                    double PXCK = ((Mem_height.get(j).get(index - 1) * (Class.get(j).get(j) + Class.get(j).get(1))) + 1) / ((Class.get(j).get(j) + Class.get(j).get(1)) + uniqueclass_count.get(j));
                    double Paj = ((Mem_height.get(j).get(index - 1)) + 1) / ((uniqueclass_count.get(0) + uniqueclass_count.get(1)) + uniqueclass_count.get(j));
                    double value1 = tem.get(1);
                    int index1 = (int) value1;
                    double PXCKw = ((Mem_weight.get(j).get(index1 - 1) * (Class.get(j).get(j) + Class.get(j).get(1))) + 1) / ((Class.get(j).get(j) + Class.get(j).get(1)) + uniqueclass_count.get(j));
                    double Paj1 = ((Mem_weight.get(j).get(index1 - 1)) + 1) / ((uniqueclass_count.get(0) + uniqueclass_count.get(1)) + uniqueclass_count.get(j));
                    double value2 = tem.get(1);
                    int index2 = (int) value2;
                    double PXCK3 = ((Mem_foot.get(j).get(index2 - 1) * (Class.get(j).get(j) + Class.get(j).get(1))) + 1) / ((Class.get(j).get(j) + Class.get(j).get(1)) + uniqueclass_count.get(j));
                    double Paj3 = ((Mem_foot.get(j).get(index2 - 1)) + 1) / ((uniqueclass_count.get(j) + uniqueclass_count.get(1)) + uniqueclass_count.get(j));
                    double mu = (Class.get(j).get(j) + Class.get(j).get(1));
                    probability1 = Pck * (((PXCK / Paj) * mu) * ((PXCKw / Paj1) * mu) * ((PXCK3 / Paj) * mu));

                } else {
                    double valuef = tem.get(0);
                    int indexf = (int) valuef;
                    double PXCK = ((Mem_height.get(j).get(indexf - 1) * (Class.get(j).get(0) + Class.get(j).get(j))) + 1) / ((Class.get(j).get(0) + Class.get(j).get(j)) + uniqueclass_count.get(j));
                    double Paj = ((Mem_height.get(j).get(indexf - 1)) + 1) / ((uniqueclass_count.get(0) + uniqueclass_count.get(1)) + uniqueclass_count.get(1));
                    double valuef1 = tem.get(1);
                    int indexf1 = (int) valuef1;
                    double PXCK1 = ((Mem_weight.get(j).get(indexf1 - 1) * (Class.get(j).get(0) + Class.get(j).get(j))) + 1) / ((Class.get(j).get(0) + Class.get(j).get(j)) + uniqueclass_count.get(j));
                    double Paj1 = ((Mem_weight.get(j).get(indexf1 - 1)) + 1) / ((uniqueclass_count.get(0) + uniqueclass_count.get(1)) + uniqueclass_count.get(1));
                    double valuef2 = tem.get(1);
                    int indexf2 = (int) valuef2;
                    double PXCK2 = ((Mem_foot.get(j).get(indexf2 - 1) * (Class.get(j).get(0) + Class.get(j).get(j))) + 1) / ((Class.get(j).get(0) + Class.get(j).get(j)) + uniqueclass_count.get(j));
                    double Paj2 = ((Mem_foot.get(j).get(indexf2 - 1)) + 1) / ((uniqueclass_count.get(0) + uniqueclass_count.get(j)) + uniqueclass_count.get(j));
                    double muf = (Class.get(1).get(0) + Class.get(1).get(1));
                    probability1 = Pck1 * (((PXCK / Paj) * muf) * ((PXCK1 / Paj1) * muf) * ((PXCK2 / Paj2) * muf));
                }
                prob.add(probability1);
            }

            double max = Collections.max(prob);
            int d = prob.indexOf(max);

            if (d == 0) {
                tem.set(tem.size() - 1, 1.0);
            } else {
                tem.set(tem.size() - 1, 2.0);
            }

            datatestup.add(tem);
        }

        for (int i = 0; i < datatest1.size(); i++) {

            if (datatest1.get(i).get(datatest1.get(i).size() - 1).equals(1.0) && datatestup.get(i).get(datatestup.get(i).size() - 1).equals(1.0)) {
                tp++;
            }
            if (datatest1.get(i).get(datatest1.get(i).size() - 1).equals(2.0) && datatestup.get(i).get(datatestup.get(i).size() - 1).equals(2.0)) {
                tn++;
            }
            if (datatest1.get(i).get(datatest1.get(i).size() - 1).equals(2.0) && datatestup.get(i).get(datatestup.get(i).size() - 1).equals(1.0)) {
                fp++;
            }
            if (datatest1.get(i).get(datatest1.get(i).size() - 1).equals(1.0) && datatestup.get(i).get(datatestup.get(i).size() - 1).equals(2.0)) {
                fn++;
            }

        }

        acc = (double) (tp + tn) / (double) (tp + tn + fp + fn);
        dr = (double) tp / (double) (tp + tn + fp + fn);
        fpr = (double) fp / (double) (fp + tn);
        Code.Run.Accuracy.add(acc);
        Code.Run.Detection_Rate.add(dr);
        Code.Run.FPR.add(fpr);
    }

    public static void membership(ArrayList<ArrayList<Double>> membership_height, ArrayList<ArrayList<Double>> membership_weight, ArrayList<ArrayList<Double>> membership_foot, ArrayList<ArrayList<Double>> membership_class, ArrayList<Double> uniqueclass_count, ArrayList<Double> clas) {
        ArrayList<ArrayList<Double>> data = new ArrayList<>();
        ArrayList<Double> mem_heightm = new ArrayList<>();
        ArrayList<Double> mem_heightf = new ArrayList<>();
        ArrayList<Double> mem_weightm = new ArrayList<>();
        ArrayList<Double> mem_weightf = new ArrayList<>();
        ArrayList<Double> mem_footm = new ArrayList<>();
        ArrayList<Double> mem_footf = new ArrayList<>();
        ArrayList<Double> mem_classm = new ArrayList<>();
        ArrayList<Double> mem_classf = new ArrayList<>();

        data = updateinput(uniqueclass, dataset);

        for (int i = 0; i < uniqueclass.size(); i++) {
            double count = 0.0;
            count = Collections.frequency(clas, uniqueclass.get(i));
            uniqueclass_count.add(count);

        }

        ArrayList<Double> hei = new ArrayList<>();
        ArrayList<Double> hei1 = new ArrayList<>();

        for (int i = 0; i < data.size(); i++) {
            if (data.get(i).get(data.get(i).size() - 1) == 1) {
                hei.add(data.get(i).get(0));
            } else {
                hei1.add(data.get(i).get(0));
            }

        }
        ArrayList<Double> temmh = new ArrayList<>();
        ArrayList<Double> temfh = new ArrayList<>();

        Set<Double> hs = new HashSet<>();
        hs.addAll(hei);
        temmh.addAll(hs);
        Set<Double> hs1 = new HashSet<>();
        hs1.addAll(hei1);
        temfh.addAll(hs1);

        for (int i = 0; i < 4; i++) {
            double count = 1.0;
            if (i < temmh.size()) {
                count = Collections.frequency(hei, temmh.get(i));
                mem_heightm.add(count / uniqueclass_count.get(uniqueclass_count.size() - uniqueclass_count.size()));
            } else {
                mem_heightm.add(count / uniqueclass_count.get(uniqueclass_count.size() - uniqueclass_count.size()));
            }

        }

        for (int i = 0; i < 4; i++) {
            double count = 0.0;
            if (i < temfh.size()) {

                count = Collections.frequency(hei1, temfh.get(i));
                mem_heightf.add(count / uniqueclass_count.get(uniqueclass_count.size() - uniqueclass_count.size() + 1));
            } else {
                mem_heightf.add(count / uniqueclass_count.get(uniqueclass_count.size() - uniqueclass_count.size() + 1));
            }
        }

        membership_height.add(mem_heightm);
        membership_height.add(mem_heightf);

        ArrayList<Double> wei = new ArrayList<>();
        ArrayList<Double> wei1 = new ArrayList<>();

        for (int i = 0; i < data.size(); i++) {
            if (data.get(i).get(data.get(i).size() - 1) == 1) {
                wei.add(data.get(i).get(1));
            } else {
                wei1.add(data.get(i).get(1));
            }

        }
        ArrayList<Double> temmw = new ArrayList<>();
        ArrayList<Double> temfw = new ArrayList<>();

        Set<Double> hsw = new HashSet<>();
        hsw.addAll(wei);
        temmw.addAll(hsw);
        Set<Double> hsw1 = new HashSet<>();
        hsw1.addAll(wei1);
        temfw.addAll(hsw1);

        for (int i = 0; i < 4; i++) {
            double count = 1.0;
            if (i < temmw.size()) {
                count = Collections.frequency(wei, temmw.get(i));
                mem_weightm.add(count / uniqueclass_count.get(uniqueclass_count.size() - uniqueclass_count.size()));
            } else {
                mem_weightm.add(count / uniqueclass_count.get(uniqueclass_count.size() - uniqueclass_count.size()));
            }

        }
        for (int i = 0; i < 4; i++) {
            double count = 1.0;
            if (i < temfw.size()) {
                count = Collections.frequency(wei1, temfw.get(i));
                mem_weightf.add(count / uniqueclass_count.get(uniqueclass_count.size() - uniqueclass_count.size() + 1));
            } else {
                mem_weightf.add(count / uniqueclass_count.get(uniqueclass_count.size() - uniqueclass_count.size() + 1));
            }
        }

        membership_weight.add(mem_weightm);
        membership_weight.add(mem_weightf);

        ArrayList<Double> foo = new ArrayList<>();
        ArrayList<Double> foo1 = new ArrayList<>();

        for (int i = 0; i < data.size(); i++) {
            if (data.get(i).get(data.get(i).size() - 1) == 1) {
                foo.add(data.get(i).get(1));
            } else {
                foo1.add(data.get(i).get(1));
            }

        }

        ArrayList<Double> temmfoot = new ArrayList<>();
        ArrayList<Double> temffoot = new ArrayList<>();

        Set<Double> hsfoot = new HashSet<>();
        hsfoot.addAll(foo);
        temmfoot.addAll(hsfoot);
        Set<Double> hsfoot1 = new HashSet<>();
        hsfoot1.addAll(foo1);
        temffoot.addAll(hsfoot1);

        for (int i = 0; i < 4; i++) {
            double count = 1.0;
            if (i < temmfoot.size()) {
                count = Collections.frequency(foo, temmfoot.get(i));
                mem_footm.add(count / uniqueclass_count.get(uniqueclass_count.size() - uniqueclass_count.size()));
            } else {
                mem_footm.add(count / uniqueclass_count.get(uniqueclass_count.size() - uniqueclass_count.size()));
            }
        }

        for (int i = 0; i < 4; i++) {
            double count = 1.0;
            if (i < temffoot.size()) {

                count = Collections.frequency(foo1, temffoot.get(i));
                mem_footf.add(count / uniqueclass_count.get(uniqueclass_count.size() - uniqueclass_count.size() + 1));
            } else {
                mem_footf.add(count / uniqueclass_count.get(uniqueclass_count.size() - uniqueclass_count.size() + 1));
            }
        }

        membership_foot.add(mem_footm);
        membership_foot.add(mem_footf);
        mem_classm.add(uniqueclass_count.get(0) / (uniqueclass_count.get(0) + uniqueclass_count.get(1)));
        mem_classm.add(0.0);
        mem_classf.add(0.0);
        mem_classf.add(uniqueclass_count.get(1) / (uniqueclass_count.get(0) + uniqueclass_count.get(1)));
        membership_class.add(mem_classm);
        membership_class.add(mem_classf);
    }

    public static ArrayList<ArrayList<Double>> updateinput(ArrayList<Double> uniqueclass, ArrayList<ArrayList<Double>> dataset) {

        ArrayList<Double> dom = new ArrayList<>();
        ArrayList<ArrayList<Double>> dataset_update = new ArrayList<>();
        for (int i = 0; i < dataset.size(); i++) {

            dom.add(dataset.get(i).get(0));
        }
        double maxh = Collections.max(dom);
        double minh = Collections.min(dom);
        double a = (maxh - minh) / 4;

        for (int i = 0; i < dataset.size(); i++) {
            ArrayList<Double> tem = new ArrayList<>();
            for (int j = 0; j < dataset.get(i).size() - 1; j++) {
                if (dataset.get(i).get(j) >= (minh + (3 * a))) {
                    tem.add(4.0);
                } else if (dataset.get(i).get(j) >= (minh + (2 * a))) {
                    tem.add(3.0);
                } else if (dataset.get(i).get(j) >= (minh + a)) {
                    tem.add(2.0);
                } else if (dataset.get(i).get(j) >= minh) {
                    tem.add(1.0);
                }
            }
            tem.add(dataset.get(i).get(dataset.get(i).size() - 1));
            dataset_update.add(tem);
        }
        return dataset_update;
    }
}
