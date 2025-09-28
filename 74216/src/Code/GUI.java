package Code;

import java.awt.Color;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import static Code.Run.Accuracy;
import static Code.Run.Detection_Rate;
import static Code.Run.FPR;

public class GUI extends javax.swing.JFrame {

    public static int task = 75, iteration = 50, gp, tr_per;
    public GUI() {
        initComponents();
    }

    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jButton1 = new javax.swing.JButton();
        jButton2 = new javax.swing.JButton();
        jScrollPane1 = new javax.swing.JScrollPane();
        jTable1 = new javax.swing.JTable();
        jLabel1 = new javax.swing.JLabel();
        jTextField1 = new javax.swing.JTextField();
        jLabel2 = new javax.swing.JLabel();
        jTextField2 = new javax.swing.JTextField();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setMinimumSize(new java.awt.Dimension(650, 450));
        getContentPane().setLayout(new org.netbeans.lib.awtextra.AbsoluteLayout());

        jButton1.setFont(new java.awt.Font("Times New Roman", 1, 14)); // NOI18N
        jButton1.setText("START");
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });
        getContentPane().add(jButton1, new org.netbeans.lib.awtextra.AbsoluteConstraints(160, 185, 120, 40));

        jButton2.setText("Run Graph");
        jButton2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton2ActionPerformed(evt);
            }
        });
        getContentPane().add(jButton2, new org.netbeans.lib.awtextra.AbsoluteConstraints(417, 80, 120, 50));

        jTable1.setModel(new javax.swing.table.DefaultTableModel(
            new Object [][] {
                {"Accuracy", null, null, null, null},
                {"Detection Rate", null, null, null, null},
                {"FPR", null, null, null, null}
            },
            new String [] {
                "", "FCM-ANN ", "ANFIS", "SVM", "Proposed SFDO-based DeepRNN"
            }
        ));
        jScrollPane1.setViewportView(jTable1);

        getContentPane().add(jScrollPane1, new org.netbeans.lib.awtextra.AbsoluteConstraints(40, 270, 540, 80));

        jLabel1.setText("Group size");
        getContentPane().add(jLabel1, new org.netbeans.lib.awtextra.AbsoluteConstraints(90, 50, 93, 28));

        jTextField1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jTextField1ActionPerformed(evt);
            }
        });
        getContentPane().add(jTextField1, new org.netbeans.lib.awtextra.AbsoluteConstraints(250, 40, 100, 40));

        jLabel2.setText("Training data (%)");
        getContentPane().add(jLabel2, new org.netbeans.lib.awtextra.AbsoluteConstraints(90, 120, 110, 24));
        getContentPane().add(jTextField2, new org.netbeans.lib.awtextra.AbsoluteConstraints(250, 110, 100, 40));

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        try {
            gp = Integer.parseInt(jTextField1.getText());       // group size
            tr_per = Integer.parseInt(jTextField2.getText());   // training percentage
            check();    
            Run.callmain();
            cloudsim.ext.gui.GuiMain.main(null);
        } 
        catch (IOException ex) {
            Logger.getLogger(GUI.class.getName()).log(Level.SEVERE, null, ex);
        } catch (Exception ex) {
            Logger.getLogger(GUI.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        
        for( int i = 0; i < Accuracy.size(); i++ ) {
            GUI.jTable1.setValueAt(Accuracy.get(i), 0, i+1);
            GUI.jTable1.setValueAt(Detection_Rate.get(i), 1, i+1);
            GUI.jTable1.setValueAt(FPR.get(i), 2, i+1); 
        }
    }//GEN-LAST:event_jButton1ActionPerformed

    private void jTextField1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jTextField1ActionPerformed
        
    }//GEN-LAST:event_jTextField1ActionPerformed

    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
        
        DefaultCategoryDataset dcd1=new DefaultCategoryDataset();
        DefaultCategoryDataset dcd2=new DefaultCategoryDataset();
        DefaultCategoryDataset dcd3=new DefaultCategoryDataset();
        
        dcd1.setValue(Accuracy.get(0),"Accuracy","FCM-ANN ");
        dcd1.setValue(Accuracy.get(1),"Accuracy","ANFIS"); 
        dcd1.setValue(Accuracy.get(2),"Accuracy","SVM");
        dcd1.setValue(Accuracy.get(3),"Accuracy","Proposed SFDO-based Deep RNN");
        
        dcd2.setValue(Detection_Rate.get(0),"Detection Rate","FCM-ANN ");
        dcd2.setValue(Detection_Rate.get(1),"Detection Rate","ANFIS"); 
        dcd2.setValue(Detection_Rate.get(2),"Detection Rate","SVM");
        dcd2.setValue(Detection_Rate.get(3),"Detection Rate","Proposed SFDO-based Deep RNN");
        
        dcd3.setValue(FPR.get(0),"FPR","FCM-ANN ");
        dcd3.setValue(FPR.get(1),"FPR","ANFIS"); 
        dcd3.setValue(FPR.get(2),"FPR","SVM");
        dcd3.setValue(FPR.get(3),"FPR","Proposed SFDO-based Deep RNN");
                
    JFreeChart jchart1 = ChartFactory.createBarChart("", "", "Accuracy", dcd1, PlotOrientation.VERTICAL,false, true, false);
    JFreeChart jchart2 = ChartFactory.createBarChart("", "", "Detection Rate", dcd2, PlotOrientation.VERTICAL,false, true, false); 
    JFreeChart jchart3 = ChartFactory.createBarChart("", "", "FPR", dcd3, PlotOrientation.VERTICAL,false, true, false);
        
    CategoryPlot plot1 = jchart1.getCategoryPlot();
    CategoryPlot plot2 = jchart2.getCategoryPlot();
    CategoryPlot plot3 = jchart3.getCategoryPlot();
    plot1.setRangeGridlinePaint(Color.white);
    plot2.setRangeGridlinePaint(Color.white);
    plot3.setRangeGridlinePaint(Color.white);
    
    ChartFrame chartFrm1 = new ChartFrame("Run - Chart 1", jchart1, true);
    ChartFrame chartFrm2 = new ChartFrame("Run - Chart 2", jchart2, true);
    ChartFrame chartFrm3 = new ChartFrame("Run - Chart 3", jchart3, true);
    chartFrm1.setVisible(true);
    chartFrm2.setVisible(true);
    chartFrm3.setVisible(true);
    chartFrm1.setSize(800,500);
    chartFrm2.setSize(800,500);
    chartFrm3.setSize(800,500);
    }//GEN-LAST:event_jButton2ActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(GUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(GUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(GUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(GUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new GUI().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton2;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JScrollPane jScrollPane1;
    private static javax.swing.JTable jTable1;
    private javax.swing.JTextField jTextField1;
    private javax.swing.JTextField jTextField2;
    // End of variables declaration//GEN-END:variables

    private void check() {
        if(iteration < 10) 
            iteration = 1;
        else
            iteration = iteration / 10;
    }
}
