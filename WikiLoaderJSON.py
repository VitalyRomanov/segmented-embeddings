import os
import json

class WikiDataLoader:
    def __init__(self, path):
 
        # Path to extracted wiki dump
        self.path = path
 
        # Prepare the list of subfolders in extracted wiki dump
        self.folders = list(filter(lambda x: os.path.isdir(os.path.join(path,x)), os.listdir(path)))
        self.folders.sort() # Ensure alphabetical order
        # List of documents from wiki dump file
        self.docs = []
        self.files = []

 
    def next_folder(self):

        if self.folders:
            c_folder = self.folders.pop(0)
            print(c_folder)

            sub_path = os.path.join(self.path, c_folder)
            self.files = list(filter(lambda x: os.path.isfile(os.path.join(sub_path,x)), os.listdir(sub_path)))
            self.files.sort()

            self.sub_path = sub_path

    def next_file(self):

        if not self.files:
            self.next_folder()

        if self.files:
            c_file = self.files.pop(0)
            print("\t{}".format(c_file))
            c_path = os.path.join(self.sub_path, c_file)

            docs = open(c_path, "r").read().split("\n")

            docs_list = []
            for doc in docs:
                if doc.strip():
                    doc_json = json.loads(doc)
                    candidate = doc_json['text'].strip()
                    if candidate:
                        docs_list.append(candidate)
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


