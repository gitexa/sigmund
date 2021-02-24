# -*- coding: utf-8 -*-
"""
Written by Evan Lalopoulos <evan.lalopoulos.2017@my.bristol.ac.uk>
University of Bristol, May 2018
Copyright (C) - All Rights Reserved
"""

import collections


class Liwc:
    """
    Class for the Linguistic Inquiry and Word Count (LIWC) dictionairy.
    The dictionary files are proprietary and can be obtained by liwc.net
    """

    def __init__(self, filepath):
        """
        :param filepath: path to the LIWC .dic file.
        """
        self.categories, self.lexicon = self._load_dict_file(filepath)
        self._trie = self._build_char_trie(self.lexicon)

    def search(self, word):
        """
        Search a word in the liwc dictionairy.
        :param word:
        :return: a list of the liwc categories the word belongs.
                 an empty list if the word is not found in the dictionary.
        """
        return self._search_trie(self._trie, word)

    def parse(self, tokens):
        """
        Parses a document and extracts raw counts of words that fall into the
        various LIWC categories.
        :param tokens: a list of tokens, a tokeniSed document
        :return: a counter with the linguistic categories found in the doc,
                and the raw count of words that fall in each category.
        """
        cat_counter = collections.Counter()

        for token in tokens:
            # Find in which categories this token falls, if any
            cats = self.search(token)
            for cat in cats:
                cat_counter[cat] += 1

        return cat_counter

    def parse_with_dict(self, tokens):
        """
        Parses a document and extracts raw counts of words that fall into the
        various LIWC categories.
        :param tokens: a list of tokens, a tokeniSed document
        :return: a counter with the linguistic categories found in the doc,
                and the raw count of words that fall in each category.
        """
        cat_counter = collections.Counter()

        for token in tokens:
            # Find in which categories this token falls, if any
            cats = self.search(token['word'])
            for cat in cats:
                cat_counter[cat] += 1

        return cat_counter

    def parse_with_index(self, tokens):
        """

        Input: tokens is a list of all tokens with (word, idx_paragraph_in_doc, idx_word_in_paragraph) to enable inverse search 
        idx_paragraph_in_doc
        idx_word_in_paragraph
        """

        cat_idx = collections.defaultdict(list)

        for token in tokens:
            # Find in which categories this token falls, if any
            cats = self.search(token['word'])
            for cat in cats:
                cat_idx[cat].append({
                    'idx_paragraph_in_doc': token['idx_paragraph_in_doc'],
                    'idx_word_in_paragraph': token['idx_word_in_paragraph']
                })
                ''' 
                (token['idx_paragraph_in_doc'],
                token['idx_word_in_paragraph']))
                '''

        return cat_idx

    def get_word_and_paragraph_by_liwc_key(
            self, liwc_keys, liwc_scores_with_index, words_with_index, paragraphs):
        """
        Inverse search - get words and corresponding paragraphs by LIWC keys using index (idx) of idx_paragraph_in_document and idx_word_in_paragraph 
        (CAVE: not very efficient, needs optimization!)
        :param tokens:
        - list with liwc_keys
        - dict with liwc_scores_with_index ('Posemo': {paragraph_in_document, word_in_paragraph})
        - dict words_with_index ({word, idx_paragraph_in_document, idx_word_in_paragraph})
        - a list with all paragraphs of the conversation (paragraph0, paragraph1, ...)
        :return: a dict with all words and paragraphs counted for the input liwc_keys
        """
        word_and_paragraph_by_liwc_key = collections.defaultdict(list)

        for key in liwc_keys:
            # print key
            # print(key)
            # get words for all indices
            for liwc_words in liwc_scores_with_index[key]:
                # get indexes
                idx_paragraph_in_doc = liwc_words['idx_paragraph_in_doc']
                idx_word_in_paragraph = liwc_words['idx_word_in_paragraph']
                # get words and origial paragraph by index
                for word in words_with_index:
                    if ((word['idx_paragraph_in_doc'] == idx_paragraph_in_doc) and (word['idx_word_in_paragraph'] == idx_word_in_paragraph)):
                        word = word['word']
                        paragraph = paragraphs[idx_paragraph_in_doc]
                        #print(f'Wort: {word} | {paragraph}')
                        word_and_paragraph_by_liwc_key[key].append((word, paragraph))
            # print('-'*10)
        return word_and_paragraph_by_liwc_key

    def _load_dict_file(self, filepath):
        liwc_file = open(filepath)

        # Key, category dict
        categories = {}

        # Word, cat_name dict
        lexicon = {}

        # '%' signals a change in the .dic file.
        # (0-1) Cats, ids
        # (>1) Words, cat_ids
        percent_sign_count = 0

        for line in liwc_file:
            stp = line.strip()

            if stp:
                parts = stp.split('\t')

                if parts[0] == '%':
                    percent_sign_count += 1
                else:
                    # If the percent sign counter equals 1, parse the LIWC
                    # categories
                    if percent_sign_count == 1:
                        categories[parts[0]] = parts[1]
                    # Else, parse lexicon
                    else:
                        lexicon[parts[0]] = [categories[cat_id]
                                             for cat_id in parts[1:]]

        return categories, lexicon

    @staticmethod
    def _build_char_trie(lexicon):
        """
        Builds a char trie, to cater for wildcard ('*') matches.
        """
        trie = {}
        for pattern, cat_names in lexicon.items():
            cursor = trie
            for char in pattern:
                if char == '*':
                    cursor['*'] = cat_names
                    break

                if char not in cursor:
                    cursor[char] = {}

                cursor = cursor[char]

            # $ signifies end of token
            cursor['$'] = cat_names

        return trie

    @staticmethod
    def _search_trie(trie, token, i=0):
        """
        Search the given char trie for paths that match the token.
        """
        if '*' in trie:
            return trie['*']
        elif '$' in trie and i == len(token):
            return trie['$']
        elif i < len(token):
            char = token[i]
            if char in trie:
                return Liwc._search_trie(trie[char], token, i + 1)
        return []
