import os
import sys


class WikiDataLoader:
    def __init__(self, path):
        # Path to extracted wiki dump
        self.path = path

        self.init()

    def init(self):
        # Prepare the list of subfolders in extracted wiki dump
        self.folders = list(filter(lambda x: os.path.isdir(os.path.join(self.path, x)), os.listdir(self.path)))
        self.folders.sort()  # Ensure alphabetical order
        # List of documents from wiki dump file
        self.docs = []

        self.files = []

    def next_folder(self):

        if self.folders:
            c_folder = self.folders.pop(0)
            print(c_folder)

            sub_path = os.path.join(self.path, c_folder)
            self.files = list(filter(lambda x: os.path.isfile(os.path.join(sub_path, x)), os.listdir(sub_path)))
            self.files = list(filter(lambda x: x[0] != '.', self.files))
            self.files.sort()

            self.sub_path = sub_path

    def next_file(self):

        if not self.files:
            self.next_folder()

        if self.files:
            c_file = self.files.pop(0)
            c_path = os.path.join(self.sub_path, c_file)
            print("\t", c_file)

            docs = open(c_path, "r").read().split("</doc>")

            docs_list = []
            for doc in docs:
                if doc.strip():
                    # filter the first line that contains <doc> tag
                    temp_doc = doc.split("\n")
                    temp_doc = list(filter(lambda x: x != '' and x[0] != "<", temp_doc))
                    # while len(temp_doc) > 0 and temp_doc[0] == "":
                    #     temp_doc.pop(0)
                    # while len(temp_doc) > 0 and temp_doc[-1] == "":
                    #     temp_doc.pop(-1)
                    docs_list.append("\n".join(temp_doc))
            self.docs = docs_list

    def next_doc(self):
        '''
        Return next available document
        '''

        if not self.docs:
            self.next_file()

        if self.docs:
            return self.docs.pop(0)
        else:
            return None

    @property
    def wiki_docs(self):
        self.init()

        while self.folders or self.files or self.docs:
            yield self.next_doc()
