package tu.dortmund.lda.data_structure;

import java.util.ArrayList;
import java.util.Collections;

public class SortedTopicList {
    private int numTopics;
    private int topicMaskSize;
    private int topicMask;

    ArrayList<Integer> encodingList;

    public SortedTopicList(int numTopics) {
        this.numTopics = numTopics;

        // find smallest topicMaskSize such that 2^topicMaskSize >= kTopics
        topicMaskSize = 1;
        while((int) Math.pow(2, topicMaskSize) < numTopics){
            topicMaskSize += 1;
        }
        topicMask = (int) (Math.pow(2, topicMaskSize) - 1);

        this.encodingList = new ArrayList<Integer>();
    }

    public void addTopic(int topic, int count) {
        encodingList.add(createEncoding(topic, count));
    }

    public void sort() {
        Collections.sort(encodingList, Collections.reverseOrder());
    }

    public int getTopic(int index){
        return recoverTopic(encodingList.get(index));
    }

    public int getCount(int index){
        return recoverWordCount(encodingList.get(index));
    }

    public int size(){
        return encodingList.size();
    }

    public void decrementTopicCount(int topic) {
        // find old topic in the array. Sadly, we need to do this iteratively
        int oldTopicIndex = -1;
        for (int i = 0; i < encodingList.size(); i++) {
            if(recoverTopic(encodingList.get(i)) == topic){
                oldTopicIndex = i;
                break;
            }
        }

        if(oldTopicIndex != -1){
            // decrement the value of the old topic by 1
            int newWordCount = recoverWordCount(encodingList.get(oldTopicIndex)) - 1;
            if(newWordCount == 0){
                encodingList.remove(oldTopicIndex);
            }
            else {
                encodingList.set(oldTopicIndex, createEncoding(topic, newWordCount));
                // make sure that array is still sorted in descending order
                for (int i = oldTopicIndex; i < encodingList.size() - 1; i++) {
                    if(encodingList.get(i) < encodingList.get(i + 1)){
                        int tmp = encodingList.get(i);
                        encodingList.set(i, encodingList.get(i + 1));
                        encodingList.set(i + 1, tmp);
                    }
                    else{
                        break;
                    }
                }
            }
        }
    }

    public void incrementTopicCount(int topic){
        // find new topic in the array. Sadly, we need to do this iteratively
        int newTopicIndex = -1;
        for (int i = 0; i < encodingList.size(); i++) {
            if(recoverTopic(encodingList.get(i)) == topic){
                newTopicIndex = i;
                break;
            }
        }

        // if new topic is not in the list yet we need to create it
        if(newTopicIndex == -1) {
            encodingList.add(createEncoding(topic, 1));
            newTopicIndex = encodingList.size()-1;
        }
        else {
            int newWordCount = recoverWordCount(encodingList.get(newTopicIndex)) + 1;
            encodingList.set(newTopicIndex, createEncoding(topic, newWordCount));
        }
        // make sure that the array is still sorted in descending order
        for (int i = newTopicIndex; i > 0; i--) {
            if(encodingList.get(i) > encodingList.get(i-1)){
                int tmp = encodingList.get(i-1);
                encodingList.set(i-1, encodingList.get(i));
                encodingList.set(i, tmp);
            }
            else{
                break;
            }
        }
    }

    private int createEncoding(int topic, int wordUsageCount){
        int encoding = wordUsageCount << topicMaskSize;
        encoding += topic;
        return encoding;
    }

    private int recoverTopic(int encoding){
        return encoding & topicMask;
    }

    private int recoverWordCount(int enconding){
        return enconding >> topicMaskSize;
    }
}
