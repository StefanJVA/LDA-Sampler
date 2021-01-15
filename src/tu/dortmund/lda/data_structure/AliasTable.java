package tu.dortmund.lda.data_structure;

import java.util.*;

public class AliasTable {
    private double[] probabilityForIndex;
    private int[] alternativeForIndex;
    private double[] unnormalizedProbability;

    private int size;
    private int sampleCount;
    private double probabilitySum;

    public AliasTable(int size) {
        this.size = size;
        this.unnormalizedProbability = new double[size];
        this.probabilityForIndex = new double[size];
        this.alternativeForIndex = new int[size];
        this.probabilitySum = 0;
    }

    public void construct() {
        double bucketSize = unnormalizedProbability.length / probabilitySum;

        // construct p such that p[i] < 1.0 if outcome i fits into an empty bucket
        double [] p = new double[size];
        for (int i = 0; i < unnormalizedProbability.length; i++) {
            p[i] = unnormalizedProbability[i] * bucketSize;
        }

        // safe smaller/bigger indexes in a single int array to avoid costly stacks or lists
        // smaller from left to right and bigger from right to left
        int[] tmpIndices = new int[size];
        int smallerCurrentIndex = -1;
        int biggerCurrentIndex = size;
        for (int i = 0; i < unnormalizedProbability.length; i++) {
            if(p[i] < 1.0){
                tmpIndices[++smallerCurrentIndex] = i;
            } else {
                tmpIndices[--biggerCurrentIndex] = i;
            }
        }

        // insert outcomes into the buckets
        while(smallerCurrentIndex >= 0 && biggerCurrentIndex < size) {
            int big = tmpIndices[biggerCurrentIndex++];
            int small = tmpIndices[smallerCurrentIndex--];

            probabilityForIndex[small] = p[small];
            alternativeForIndex[small] = big;

            p[big] -= (1.0 - p[small]);

            if (p[big] < 1.0) {
                tmpIndices[++smallerCurrentIndex] = big;
            } else {
                tmpIndices[--biggerCurrentIndex] = big;
            }
        }
        while(biggerCurrentIndex < size) {
            probabilityForIndex[tmpIndices[biggerCurrentIndex++]] = 1.0;
        }
        while (smallerCurrentIndex >= 0) {
            probabilityForIndex[tmpIndices[smallerCurrentIndex--]] = 1.0;
        }
    }

    public int sample(SplittableRandom random){
        sampleCount++;

        int randomBucket = random.nextInt(this.size);
        if(probabilityForIndex[randomBucket] > random.nextDouble()){
            return randomBucket;
        } else {
            return alternativeForIndex[randomBucket];
        }
    }

    public void resetSampleCount() {
        this.sampleCount = 0;
    }

    public int getSampleCount() {
        return this.sampleCount;
    }

    public double[] getUnnormalizedProbability() {
        return this.unnormalizedProbability;
    }

    public void setUnnormalizedProbability(double[] unnormalizedProbability) {
        if (unnormalizedProbability.length != this.size) {
            throw new IllegalArgumentException("Different lengths: " + unnormalizedProbability.length + "!=" + this.size);
        }
        this.unnormalizedProbability = unnormalizedProbability;
    }

    public double getProbabilitySum() {
        return this.probabilitySum;
    }

    public void setProbabilitySum(double probabilitySum) {
        this.probabilitySum = probabilitySum;
    }


}
