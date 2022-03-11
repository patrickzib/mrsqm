
#include <vector>
#include <map>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <fstream>
#include <set>

#include "common.h"

#define NULL nullptr

using namespace std;

// a trie data structure for fast matching subsequences


struct Feature
{
    int feature_index;
    vector<string> subsequences;
};

class FNode
{
private:
    
	void initialize_fnode()
	{		
		feature_index = -1;
        
	}

public:
	int feature_index;
	map<char, FNode *> children;

	vector<Feature> features;

	FNode()
	{
		initialize_fnode();
	}

	~FNode()
	{
		for (map<char, FNode *>::iterator itr = children.begin(); itr != children.end(); itr++)
		{
			delete itr->second;
		}		
	}

	

	FNode *get_child(char c)
	{
		if (children[c] == NULL)
		{
			children[c] = new FNode();
			// children[c]->ngram = ngram + c;
		}
		return children[c];
	}

	FNode *get_child_without_creating(char c)
	{
		return children[c];
	} 
};

class SeqTrie
{
private:
    FNode * root;
    int num_of_features;

    void build(vector<string> sequences)
    {

        num_of_features = sequences.size();
        root = new FNode();
        for (int i = 0; i < sequences.size(); i++){
            string s = sequences[i];
            
            FNode* current_node = root;
            Feature new_feature;
            new_feature.feature_index = i;
            new_feature.subsequences = split_string(s," ");
            //cout << "New Feature: " << endl;
            //for (string tmp: new_feature.subsequences){
            //    cout << tmp << endl;
            //}

            for (int ii = 0; ii < s.length();ii++){
                if (s.at(ii) == ' '){  // only build trie from the first channel                   
                    break;
                }
                current_node = current_node->get_child(s.at(ii));
            } 
            current_node->features.push_back(new_feature);
            //cout << "Add new feature. " << endl;                      
        }
    }

    

public:
    SeqTrie(vector<string> sequences){
        
        build(sequences);
    }

    vector<int> multivariate_search(vector<string> mv_sequence){
        vector<int> count(num_of_features,0);

        vector<FNode*> current_nodes;
        current_nodes.push_back(root);

        for (int i = 0;i < mv_sequence[0].length();i++) {

            char c = mv_sequence[0].at(i);            
            vector<FNode*> next_nodes;
            next_nodes.push_back(root);

            for (FNode* n: current_nodes){
                FNode* next = n->get_child_without_creating(c);
                if (next != NULL){
                    // if (next->feature_index >= 0){
                    //     count[next->feature_index]++;
                    // }
                    if (next->features.size() > 0){
                        //cout << "First channel matched." << endl;
                        for (Feature f: next->features){
                            int subseq_length = f.subsequences[0].length();
                            int subseq_start = i - subseq_length + 1;
                            bool matched = true;
                            for(int ii = 1; ii < f.subsequences.size();ii++){ //ignore the first channel
                                //cout << "Next Channel: " << f.subsequences[ii] << " vs " << mv_sequence[ii].substr(subseq_start,subseq_length) << endl;
                                if (f.subsequences[ii] != mv_sequence[ii].substr(subseq_start,subseq_length)){
                                    
                                    matched = false;
                                    break;
                                }
                            }
                            if (matched){
                                count[f.feature_index]++;
                            }
                        }
                    }
                    next_nodes.push_back(next);
                }
            }

            current_nodes = next_nodes;



        }

        return count;


    }   

    vector<int> search(string sequence){
        vector<int> count(num_of_features,0);

        vector<FNode*> current_nodes;
        current_nodes.push_back(root);

        for (char c: sequence){
            
            vector<FNode*> next_nodes;
            next_nodes.push_back(root);

            for (FNode* n: current_nodes){
                FNode* next = n->get_child_without_creating(c);
                if (next != NULL){
                    if (next->feature_index >= 0){
                        count[next->feature_index]++;
                    }
                    
                    next_nodes.push_back(next);
                }
            }

            current_nodes = next_nodes;



        }

        return count;


    }



};