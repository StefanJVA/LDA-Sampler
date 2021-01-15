package tu.dortmund.lda;

public interface LdaModel {
    public void initialize();
    public void run(int iterations);
    public double[][] getTheta();
    public double[][] getPhi();
    public int[][] getMatZ();
    public void setSeed(long seed);
    public void setAlpha(double[] alpha);
    public void setAlpha(double alpha);
    public void setBeta(double[] beta);
    public void setBeta(double beta);
    public double[] getAlpha();
    public double[] getBeta();
    public int getNumNormalTopics();
    public int getNumTotalTopics();
    public double getLogLikelihood();
    public int[] getAssignmentsPerTopic();
    public int getNumTokens();
    public int getVocabularySize();

    default double log2(double a) {
        return Math.log(a) / Math.log(2);
    }

    default void showProgressInConsole(int percentage) {
        System.out.print(getClass().getSimpleName() + ": Processing... " + percentage + "% " +"\r");
    }

    public enum BTopicInitMethod {
        TF,
        DF,
        TFxDF,
        TF_SQUARE,
        TFxDF_SQUARE,
        Uniform
    }

    public enum WordWeightInitMethod {
        Normal,
        PMI,
        LogEntropy,
        InformationContent
    }
}
