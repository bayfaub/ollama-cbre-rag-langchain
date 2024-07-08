def start_app():
    print("RAG AI project starting...")
    run = True
    while run:
    

        print("This GenAI application can answer questions relating to data in CBRE's knowledge base!(Enter -1 to exit)")
        query = input("Please input a real estate related question or enter -1 to exit the application:\n")
        
        if int(query) == -1:
            run = False
            exit()



if __name__ == "__main__":
    start_app()