package tu.dortmund.lda;

public interface LdaModel {
    /**
     * This method allocates the necessary memory and data structures in order to
     * run the model. It needs to be called before the model can be run!
     */
    public void initialize();

    /**
     * Let the model run its sampling algorithm. One iteration is one full corpus
     * sweep.
     * 
     * @param iterations Number of iterations that should be executed.
     */
    public void run(int iterations);

    public double[][] getTheta();

    public double[][] getPhi();

    public int[][] getMatZ();

    public int getNumTopics();

    public int getNumTokens();

    public int getVocabularySize();

    public double[] getAlpha();

    public double[] getBeta();

    public double getLogLikelihood();

    public void setSeed(long seed);

    /**
     * Set a asymmetric alpha prior.
     * 
     * @param alpha Array of prior values. The index of each value correspond the
     *              topic number. Array length should therefore be equal to the
     *              number of topics.
     */
    public void setAlpha(double[] alpha);

    /**
     * Set a symmetric alpha prior.
     * 
     * @param alpha Alpha prior value.
     */
    public void setAlpha(double alpha);

    /**
     * Set a asymmetric beta prior.
     * 
     * @param alpha Array of prior values. The index of each value correspond the
     *              term number. Array length should therefore be equal to the
     *              vocabulary size.
     */
    public void setBeta(double[] beta);

    /**
     * Set a symmetric beta prior.
     * 
     * @param alpha Beta prior value.
     */
    public void setBeta(double beta);

    default double log2(double a) {
        return Math.log(a) / Math.log(2);
    }
}
