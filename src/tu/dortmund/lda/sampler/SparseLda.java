package tu.dortmund.lda.sampler;

import tu.dortmund.lda.data_structure.SortedTopicList;

public class SparseLda extends GibbsLda {

    private SortedTopicList[] nonzeroTopicTerm;
    protected double ssum;
    protected double[] qDoc;

    public SparseLda(int[][] documents, int vocabularySize, int k) {
        super(documents, vocabularySize, k);
    }

    @Override
    public void initialize() {
        super.initializeCountMatrices();

        // create data structure to find for each word the topics that assigns it at least once
        // sort topics in descending order by how often the assign the word
        nonzeroTopicTerm = new SortedTopicList[vocabularySize];
        for (int w = 0; w < vocabularySize; w++) {
            SortedTopicList list = new SortedTopicList(numTotalTopics);
            for (int t = 0; t < numTotalTopics; t++) {
                if(matTopicWord[t][w] > 0.0){
                    list.addTopic(t, (int)(matTopicWord[t][w]+0.1));
                }
            }
            list.sort();
            nonzeroTopicTerm[w] = list;
        }

        // compute the ssum bucket and the part of qDocSpecific that is not document specific
        qDoc = new double[numTotalTopics];
        ssum = 0.0;
        for (int t = 0; t < numTotalTopics; t++) {
            qDoc[t] = alpha[t] / (vecTopic[t] + betaSum);
            ssum += qDoc[t];
        }
    }

    @Override
    protected void fullCorpusSweep() {
        double[] q = new double[numNormalTopics];

        for (int document = 0; document < documents.length; document++) {
            // compute document specific bucket rsum and add document specific information to qDocSpecific
            double rsum = 0.0;
            for (int t = 0; t < numNormalTopics; t++) {
                double tmp = matDocTopic[document][t] / (vecTopic[t] + betaSum);
                rsum += tmp;
                qDoc[t] += tmp;
            }

            for (int token = 0; token < documents[document].length; token++) {
                int word = documents[document][token];
                int topic = matZ[document][token];

                rsum *= beta[word];
                ssum *= beta[word];

                super.decrementCountMatrices(document, word, topic);

                // update bucket sums (since we have just decremented the count matrices)
                double alphaTimesBeta = alpha[topic] * beta[word];
                double denominator = betaSum + vecTopic[topic];
                ssum -= alphaTimesBeta / (denominator + 1);
                ssum += alphaTimesBeta / denominator;
                rsum -= ((matDocTopic[document][topic] + 1) * beta[word]) / (denominator + 1);
                rsum += (matDocTopic[document][topic] * beta[word]) / denominator;
                qDoc[topic] = (alpha[topic] + matDocTopic[document][topic]) / denominator;

                // compute qsum bucket
                double qsum = 0.0;
                SortedTopicList nonzeroTopics = nonzeroTopicTerm[word];
                for (int i = 0; i < nonzeroTopics.size(); i++) {
                    int encodedTopic = nonzeroTopics.getTopic(i);
                    int encodedWordCount = nonzeroTopics.getCount(i);
                    // decrement wordCount by 1 if encodedTopic is the current topic
                    if(encodedTopic == topic && encodedWordCount > 0) {
                        encodedWordCount--;
                    }
                    q[i] = encodedWordCount * qDoc[encodedTopic];
                    qsum += q[i];
                }

                double normalizingConstant = ssum + rsum + qsum;

                // sample new topic assignment with the sparse lda method
                double u = random.nextDouble() * normalizingConstant;
                if(u < ssum) {
                    u /= beta[word];
                    for (int t = 0; t < numNormalTopics; t++) {
                        u -= alpha[t] / (vecTopic[t] + betaSum);
                        if(u <= 0){
                            topic = t;
                            break;
                        }
                    }
                }
                else if(u < (ssum + rsum)) {
                    u -= ssum;
                    u /= beta[word];
                    for (int t = 0; t < numNormalTopics; t++) {
                        u -= matDocTopic[document][t] / (vecTopic[t] + betaSum);
                        if(u <= 0){
                            topic = t;
                            break;
                        }
                    }
                }
                else {
                    u -= ssum + rsum;
                    for (int i = 0; i < q.length; i++) {
                        u -= q[i];
                        if(u <= 0){
                            topic = nonzeroTopics.getTopic(i);
                            break;
                        }
                    }
                }


                // update the wordTopicEncodingMatrix if the topic has changed
                if(topic != matZ[document][token]){
                    nonzeroTopics.decrementTopicCount(matZ[document][token]);
                    nonzeroTopics.incrementTopicCount(topic);
                }

                // update bucket sums with the new topic assignment
                alphaTimesBeta = alpha[topic] * beta[word];
                denominator = betaSum + vecTopic[topic];
                ssum -= alphaTimesBeta / denominator;
                ssum += alphaTimesBeta / (denominator + 1);
                rsum -= (matDocTopic[document][topic] * beta[word]) / denominator;
                rsum += ((matDocTopic[document][topic] + 1) * beta[word]) / (denominator + 1);
                qDoc[topic] = (alpha[topic] + matDocTopic[document][topic] + 1) / (denominator + 1);


                super.incrementCountMatrices(document, word, topic);

                matZ[document][token] = topic;

                rsum /= beta[word];
                ssum /= beta[word];
            }

            // remove document specific information from qDocSpecific
            for (int t = 0; t < numNormalTopics; t++) {
                qDoc[t] -= matDocTopic[document][t] / (vecTopic[t] + betaSum);
            }
        }
    }
}
