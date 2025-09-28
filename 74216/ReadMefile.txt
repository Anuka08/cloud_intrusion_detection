1. PROJECT TITLE
	Keywords: big data, mapreduce, collaborative filtering, sentiment analysis, recommendation.

2. HARDWARE REQUIREMENTS
	OS-Windows 10
	RAM-8GB
	ROM-More than 100 GB
	GPU-Yes
	CPU-1.7 Ghz

3. SOFTWARE REQUIREMENTS
	Software name(JAVA): jdk 1.8  
	(Download link: https://www.oracle.com/in/java/technologies/javase/javase-jdk8-downloads.html)

	Software name: Netbeans: Version: 8.2 
	(Download link: https://netbeans.org/downloads/old/8.2/)

4. HOW TO RUN
	Step 1: Loading the project in NETBEANS
	 => Open Netbeans 8.2
	 => Go to File, select Open Project, browse the project from your drive and select it.(Project will get loaded into the netbeans)
	 => For the first time, netbeans will take some time to load the settings.(Please wait if any process is loading on the bottom of the screen)
	Step 2: Run the program and getting the results 
	 => Open the Project (Source Packages-> code-> GUI.java). 
	 => Open ‘GUI.java’ and click Run button (or click Source in top, right click over the screen, select Run File).
	 => GUI window will open,
		1) Enter Group size (4)
		2) Enter Training data(%) (80)
		3) Click START, after some time results will be displayed.
		4) Expected Execution time: 30 minutes

5. IMPORTANT JAVA FILE AND DESCRIPTION:
	code-> GUI.java: User Interface, code starts here
	code-> Run.java: Main code
	code-> FCM.java: Clustering/ grouping data
	Proposed_SFDO_DRNN-> Feature_fusion.java: Feature fusion
	Proposed_SFDO_DRNN-> SailFish_update.java: SailFish+Dolphin optimization
		Update equation: line 109
	Proposed_SFDO_DRNN-> SFDO_fitness.java: Objective function (loss from DRNN)
	Proposed_SFDO_DRNN-> DRNN.java: DRNN classification
	
