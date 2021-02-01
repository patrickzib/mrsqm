
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
    
	void initialize_fnode()
	{		
		feature_index = -1; // not a feature
        freq = 0;
        ngram = "";
        
	}

public:
	int feature_index;
    int freq;
	map<char, FNode *> children;
    vector<pair<int,int>> locs;
    string ngram;

	

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
			children[c]->ngram = ngram + c;
		}
		return children[c];
	}

	FNode *get_child_without_creating(char c)
	{
		return children[c];
	} 

    void print_r()
	{


		for (map<char, FNode *>::iterator itr = children.begin(); itr != children.end(); itr++)
		{
            cout << itr->first << " {" << endl;
			itr->second->print_r();
            cout << "}" << endl;
		}
        
	}
};

class SeqTrie
{
private:
    FNode * root;
    int num_of_features;
    int size;

    void build(vector<string> sequences)
    {
        root = new FNode();
        for (int i = 0; i < sequences.size(); i++){
            string s = sequences[i];
            FNode* current_node = root;
            for (char c: s){
                current_node = current_node->get_child(c);
            }
            current_node->feature_index = i;
            current_node->freq++;
        }
    }

    void build_subword_trie(vector<string> sequences)
    {
        root = new FNode(); // prepare inverted index for all characters
        for (int i = 0; i < sequences.size();i++){
            for (int j = 0; j < sequences[i].length(); j++){
                root->get_child(sequences[i].at(j))->locs.push_back(make_pair(i,j));            
            }
        }

        vector<FNode*> unvisited_nodes; // initialize unvisited
        for (map<char, FNode *>::iterator itr = root->children.begin(); itr != root->children.end(); itr++)
		{
			unvisited_nodes.push_back(itr->second);
		}

        while(!unvisited_nodes.empty()){
            FNode* cn = unvisited_nodes.back();
            unvisited_nodes.pop_back();
            //cout << "Add node: " << cn->ngram << endl;
            size++;

            for (auto loc: cn->locs){ // expand node
                if (loc.second < (sequences[loc.first].length() - 1)){ // check if it has reached the end of the sequence
                    char c = sequences[loc.first].at(loc.second + 1); // next char
                    cn->get_child(c)->locs.push_back(make_pair(loc.first,loc.second + 1));
                }
            }
            cn->locs.clear(); //won't need the locations again

            for (map<char, FNode *>::iterator itr = cn->children.begin(); itr != cn->children.end(); itr++) // add new nodes to unvisited
		    {
			    unvisited_nodes.push_back(itr->second);
		    }
        }

    }

    

public:

    SeqTrie(){     
        size = 0;   
    }

    SeqTrie(vector<string> sequences){
        num_of_features = sequences.size();
        size = 0;
        build(sequences);
    }

    SeqTrie(vector<string> sequences, bool subword){
        num_of_features = sequences.size();
        size = 0;
        if (subword){
            //cout << "Build subword trie" << endl;
            build_subword_trie(sequences);
            // cout << "Size: " << this->get_size() << endl;
        } else {
            build(sequences);
        }
        //root->print_r();
        
    }

    ~SeqTrie(){
        delete this->root;
    }

    int get_size(){ // size excluding root
        return size;
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

    double euc_length(){
        vector<FNode*> unvisited;
        double sosq = 0.0;
        unvisited.push_back(this->root);
        while(!unvisited.empty()){
            FNode* cn = unvisited.back();
            unvisited.pop_back();

            if (cn->children.empty()){ // on leaf node
                sosq += cn->freq * cn->freq;
            } else { // keep going
                for (map<char, FNode *>::iterator lit = cn->children.begin(); lit != cn->children.end(); lit++){
                    unvisited.push_back(lit->second);
                }
            }

        }

        return sqrt(sosq);
    }

    double cosine_similarity(SeqTrie& t){
        vector<pair<FNode*,FNode*>> current_pairs;
        current_pairs.push_back(make_pair(this->root, t.root));

        double dot_product = 0.0;

        while(!current_pairs.empty()){
            pair<FNode*,FNode*> p = current_pairs.back();
            current_pairs.pop_back();

            if (p.first->children.empty() && p.second->children.empty()){ // found a match
               dot_product += p.first->freq * p.second->freq; 
            }

            for (map<char, FNode *>::iterator lit = p.first->children.begin(); lit != p.first->children.end(); lit++){
			    for (map<char, FNode *>::iterator rit = p.second->children.begin(); rit != p.second->children.end(); rit++){
                    if (lit->first == rit->first){ 
                        current_pairs.push_back(make_pair(lit->second,rit->second));
                        break; // shouldn't be anymore match
                    }
                }
		    }

        }
        
        return dot_product / (this->euc_length() * t.euc_length());
    }

    double cosine_similarity_subword(SeqTrie& t){
        vector<pair<FNode*,FNode*>> current_pairs;
        current_pairs.push_back(make_pair(this->root, t.root));

        double dot_product = -1; //to discount root

        while(!current_pairs.empty()){
            pair<FNode*,FNode*> p = current_pairs.back();
            current_pairs.pop_back();

            // cout << "Match: " << p.first->ngram << endl;
            dot_product += 1;

            for (map<char, FNode *>::iterator lit = p.first->children.begin(); lit != p.first->children.end(); lit++){
			    for (map<char, FNode *>::iterator rit = p.second->children.begin(); rit != p.second->children.end(); rit++){
                    if (lit->first == rit->first){ 
                        current_pairs.push_back(make_pair(lit->second,rit->second));
                        break; // shouldn't be anymore match
                    }
                }
		    }

        }       
        
        return dot_product / sqrt(this->get_size() * t.get_size());
    }



};

double cosine_similarity(vector<string> & first_seqs, vector<string> & second_seqs){

    SeqTrie t1 (first_seqs,true);
    SeqTrie t2 (second_seqs,true);
    return 1.0 - t1.cosine_similarity_subword(t2);
};

double cosine_similarity_with_set(vector<string> & first_seqs, vector<string> & second_seqs){

    double dot_product = 0.0;

    set<string> dict(first_seqs.begin(),first_seqs.end());

    for (string s: second_seqs){
        if (dict.find(s) != dict.end()){
            dot_product += 1.0;
        }
    }

    return 1.0 - dot_product / sqrt(first_seqs.size() * second_seqs.size());
    

    
    
};