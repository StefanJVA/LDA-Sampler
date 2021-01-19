package tu.dortmund.lda.sampler;

import tu.dortmund.lda.data_structure.FPlusTree;

import java.util.ArrayList;

public class FTreeLda extends GibbsLda {

    class DocToken {
        private int documentID;
        private int tokenID;

        public DocToken(int documentID, int tokenID) {
            this.documentID = documentID;
            this.tokenID = tokenID;
        }

        public int getDocumentID() {
            return documentID;
        }

        public int getTokenID() {
            return tokenID;
        }
    }

    private ArrayList<Integer>[] nonzeroDocTopic;
    private ArrayList<DocToken>[] wordOccurences;

    public FTreeLda(int[][] documents, int vocabularySize, int k) {
        super(documents, vocabularySize, k);
    }

    @Override
    public void initialize() {
        super.initializeCountMatrices();

        // build up nonzero documents data structure
        nonzeroDocTopic = new ArrayList[documents.length];
        for (int document = 0; document < documents.length; document++) {
            ArrayList<Integer> nonzeroList = new ArrayList<Integer>();
            for (int topic = 0; topic < numTopics; topic++) {
                if(matDocTopic[document][topic] > 0 && !nonzeroList.contains(topic)) {
                    nonzeroList.add(topic);
                }
            }
            nonzeroDocTopic[document] = nonzeroList;
        }

        /*
         * Build up a data structure that contains the occurences of each term in the
         * corpus. This makes it possible to iterate through the corpus term by term
         * instead of word token by word token. The data structure consumes a lot of
         * memory and could be avoided if the corpus representation would be different.
         * However for the sake of simplicity this data structure is used.
         */
        wordOccurences = new ArrayList[vocabularySize];
        for (int word = 0; word < vocabularySize; word++) {
            wordOccurences[word] = new ArrayList<DocToken>();
        }
        for (int document = 0; document < documents.length; document++) {
            for (int token = 0; token < documents[document].length; token++) {
                int word = documents[document][token];
                DocToken dt = new DocToken(document, token);
                wordOccurences[word].add(dt);
            }
        }

    }

    @Override
    protected void fullCorpusSweep() {
        double[] p = new double[numTopics];
        double[] q = new double[numTopics];
        double[] q2 = new double[numTopics];

        for (int word = 0; word < wordOccurences.length; word++) {
            for (int topic = 0; topic < numTopics; topic++) {
                q2[topic] = (matTopicWord[topic][word] + beta[word]) / (vecTopic[topic] + betaSum);
                q[topic] = alpha[topic] * q2[topic];
            }
            FPlusTree ftree = new FPlusTree(q.length);
            ftree.build(q);
            FPlusTree fTree = ftree;

            for (int occurrence = 0; occurrence < wordOccurences[word].size(); occurrence++) {
                DocToken dt = wordOccurences[word].get(occurrence);
                int document = dt.documentID;
                int token = dt.tokenID;
                int topic = matZ[document][token];

                super.decrementCountMatrices(document, word, topic);
                if(matDocTopic[document][topic] == 0){
                    nonzeroDocTopic[document].remove(Integer.valueOf(topic));
                }

                q2[topic] = (matTopicWord[topic][word] + beta[word]) / (vecTopic[topic] + betaSum);
                fTree.update(topic, alpha[topic] * q2[topic]);

                ArrayList<Integer> nonzeroTopics = nonzeroDocTopic[document];
                double pSum = 0.0;
                for (int i = 0; i < nonzeroDocTopic[document].size(); i++) {
                    int t = nonzeroTopics.get(i);
                    pSum += matDocTopic[document][t] * q2[t];
                    p[i] = pSum;
                }

                double qSum = fTree.getProbabilitySum();

                double u = random.nextDouble() * (pSum + qSum);
                if(u < pSum){
                    int index = lowerBound(p, nonzeroTopics.size(), u) ;
                    topic = nonzeroTopics.get(index);
                } else {
                    topic = fTree.sample(random);
                }

                if(matDocTopic[document][topic] == 0){
                    nonzeroDocTopic[document].add(topic);
                }
                super.incrementCountMatrices(document, word, topic);

                q2[topic] = (matTopicWord[topic][word] + beta[word]) / (vecTopic[topic] + betaSum);
                fTree.update(topic, alpha[topic] * q2[topic]);

                matZ[document][token] = topic;
            }
        }
    }
}
