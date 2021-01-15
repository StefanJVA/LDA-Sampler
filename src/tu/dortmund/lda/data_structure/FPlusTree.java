package tu.dortmund.lda.data_structure;

import java.util.SplittableRandom;

public class FPlusTree {

    private int size;
    private double[] treeNodes;

    public FPlusTree(int size) {
        this.size = size;
        this.treeNodes = new double[2 * this.size];
    }

    public void build(double[] weights) {
        if (weights.length != this.size) {
            System.err.println("Provided weights array does not match the tree size");
        }

        // init leafs
        for (int i = 2 * size - 1; i >= size; i--) {
            treeNodes[i] = weights[i - size];
        }
        // init inner nodes
        for (int i = size - 1; i > 0; i--) {
            treeNodes[i] = treeNodes[2 * i] + treeNodes[2 * i + 1];
        }
    }

    public void update(int i, double newWeight) {
        int j = i + size;
        double delta = newWeight - treeNodes[j];
        treeNodes[j] = newWeight;
        j >>= 1;
        while(j > 0) {
            treeNodes[j] += delta;
            j >>= 1;
        }
    }

    public int sample(SplittableRandom random) {
        int i = 1;
        double u = random.nextDouble() * treeNodes[i];
        while(i < size) {
            i <<= 1;
            if(u >= treeNodes[i]) u -= treeNodes[i++];
        }
        return i - size;
    }

    public double getProbabilitySum() {
        return treeNodes[1];
    }

    public double getLeafWeight(int i) {
        return treeNodes[i+size];
    }

    public void setTreeNodes(double[] treeNodes) {
        this.treeNodes = treeNodes;
    }
}
