
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

class FNode
{
private:
    
	
public:	
	map<char, FNode *> children;
    vector<int> docs;

	

	FNode()
	{
		
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
        root = new FNode();
        for (int i = 0; i < sequences.size(); i++){
            string s = sequences[i];
            FNode* current_node = root;
            for (char c: s){
                current_node = current_node->get_child(c);
                current_node->docs.push_back(i);
            }
            
        }
    }

    

public:
    SeqTrie(vector<string> sequences){
        num_of_features = sequences.size();
        build(sequences);
    }

    vector<vector<double>> match(vector<string> seqs){
        vector<vector<double>> fm;
        for (string s: seqs){
            
            FNode* current_node = root;
            for (char c:s){

            }
        }

        return fm;
    }

    

    




};