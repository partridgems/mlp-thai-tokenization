# -*- mode: Python; coding: utf-8 -*-

"""For the purposes of classification, a corpus is defined as a collection
of labeled documents. Such documents might actually represent words, images,
etc.; to the classifier they are merely instances with features."""

from abc import ABCMeta, abstractmethod
from csv import reader as csv_reader
from glob import glob
import json
from os.path import basename, dirname, split, splitext

class Sequence(object):

    def __init__(self, document_list):
        self.sequence = document_list
        self.label = [x.label for x in document_list]

    # Act as a mutable container for documents in the sequence
    def __len__(self): return len(self.sequence)
    def __iter__(self): return iter(self.sequence)
    def __getitem__(self, key): return self.sequence[key]
    def __setitem__(self, key, value): self.sequence[key] = value
    def __delitem__(self, key): del self.sequence[key]

class Document(object):
    """A document completely characterized by its features.func_closure

    feature_vector is a vector of feature indices.
    For CRF implementation, we cannot use string to represent features anymore.
    """

    max_display_data = 10 # limit for data abbreviation

    def __init__(self, data, label=None, source=None):
        self.data = data
        self.label = label
        self.source = source
        self.feature_vector = []

    def __repr__(self):
        return ("<%s: %s>" % (self.label, self.abbrev()) if self.label else
                "%s" % self.abbrev())

    def abbrev(self):
        return (self.data if len(self.data) < self.max_display_data else
                self.data[0:self.max_display_data] + "...")

    def features(self):
        """A list of features that characterize this document."""
        return [self.data]

class Character(Document):
    """This featurization should get at least 82% accuracy (vs 79% baseline)
    on the development set

    Use current type, t-1 type, t+1 type
    """
    def sequence_features(self, current_time_step, sequence):
        features = ['T0=%s' % sequence[current_time_step].data[1]]
        features.append('**BIAS TERM**')
        if current_time_step == 0:
            features.append('T-1=START')
        else:
            features.append('T-1=%s' % sequence[current_time_step-1].data[1])

        if current_time_step == (len(sequence)-1):
            features.append('T+1=END')
        else:
            features.append('T+1=%s' % sequence[current_time_step+1].data[1])
        return features

class Character2(Document):
    """This featurization should get at least 87% accuracy (vs 79% baseline)
    on the development set

    Use current, t-2, t-1, t+1, t+2 types
    and current, t-2, t-1, t+1, t+2 characters
    """

    def sequence_features(self, current_time_step, sequence):
        features = ['T0=%s' % sequence[current_time_step].data[1],
                'T0=%s' % sequence[current_time_step].data[0]]
        features.append('**BIAS TERM**')
        for i in range(1, 2+1):
            if (current_time_step + i) >= len(sequence):
                features.append('T+%s=END' % i)
            else:
                features.append('T+%s=%s' % (i, sequence[current_time_step+i].data[0]))
                features.append('T+%s=%s' % (i, sequence[current_time_step+i].data[1]))

            if (current_time_step - i) < 0:
                features.append('T-%s=START' % i)
            else:
                features.append('T-%s=%s' % (i, sequence[current_time_step-i].data[0]))
                features.append('T-%s=%s' % (i, sequence[current_time_step-i].data[1]))
        return features

class CharacterTest(Document):
    """This featurization should get you close to 100%
    because we are using the true label as features

    """
    def sequence_features(self, current_time_step, sequence):
        features= []
        features.append(sequence[current_time_step].label)
        return features

class Corpus(object):
    """An abstract collection of documents."""

    __metaclass__ = ABCMeta

    def __init__(self, datafiles, document_class=Document):
        self.documents = []
        self.datafiles = glob(datafiles)
        for datafile in self.datafiles:
            self.load(datafile, document_class)

    # Act as a mutable container for documents.
    def __len__(self): return len(self.documents)
    def __iter__(self): return iter(self.documents)
    def __getitem__(self, key): return self.documents[key]
    def __setitem__(self, key, value): self.documents[key] = value
    def __delitem__(self, key): del self.documents[key]

    @abstractmethod
    def load(self, datafile, document_class):
        """Make labeled document instances for the data in a file."""
        pass

class ThaiWordCorpus(Corpus):

    def load(self, datafile, document_class=Character):
        with open(datafile) as file:
            sequence = []
            line_number = 0
            for line in file:
                character, char_type, tag = line.strip().split(' ')
                if character == 'EOS' or tag == 'O':
                    if len(sequence) > 0:
                        self.documents.append(Sequence(sequence))
                    sequence = []
                else:
                    sequence.append(document_class((character, char_type), tag, line_number))
                line_number += 1
        self.featurize()

    def featurize(self):
        self.label_codebook = {}
        self.feature_codebook = {}
        for sequence in self.documents:
            for t, document in enumerate(sequence):
                features = document.sequence_features(t, sequence)
                for feature in features:
                    if feature not in self.feature_codebook:
                        self.feature_codebook[feature] = len(self.feature_codebook)
                    document.feature_vector.append(self.feature_codebook[feature])
                if document.label not in self.label_codebook:
                   self.label_codebook[document.label] = len(self.label_codebook)
                document.label_index = self.label_codebook[document.label]
