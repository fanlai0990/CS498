# CS 498: Machine Learning Systems (F'26)

## Logistics
**Lectures**: 1310 Digital Computer Laboratory, MW: 9:30 AM – 10:45 AM
| Member (NetID) | Role | Office Hours |
| :---------------- | :--- | :----------- |
| [Fan Lai](https://fanlai.me/) (fanlai) | Instructor | 3128 Siebel Center. W 3:00 PM – 4:00 PM
| TBD | TAs |  [Zoom](). F 2:00 PM - 3:00 PM

**Canvas**:  *ALL* communication regarding this course must be via [Canvas](https://canvas.illinois.edu/courses/73559). This includes questions, discussions, announcements, assignments, as well as private messages.

## Course Description
**Learning Objectives**: This course will introduce the basic concepts and cutting-edge practices in the design and implementation of efficient software systems for supporting machine learning (ML) models, with a particular focus on Generative AI (GenAI). By the end of the course, students will be able to:

-   Understand and critique the design principles behind state-of-the-art ML systems, from model architecture to system-level considerations.
-   Develop and utilize tools to profile, monitor, and optimize the performance of ML systems
-   Explore and conduct research in topics related to the practical deployment and optimization of ML systems, contributing to the evolving landscape of efficient ML operations.

**Structure**: The course will combine lectures, guest lectures from practioners, lab assignments, reading summaries, and a semester-long project. We will explore key ML topics from a systems perspective, addressing the relevant challenges across the ML lifecycle. Topics include, but are not limited to:
- Basics of ML models from a systems perspective; 
- Systems for ML lifecycle (pre-training, training, fine-tuning, inference serving, and grounding); 

Note that this course is **NOT focused on AI methods**.  Instead, we will *focus on how one can build software systems* so that existing AI methods can be used in practice and new AI methods can emerge. 

**Prerequisites**:  Students are expected to have good programming skills and must have taken at least one systems-related course (from operating systems, databases, distributed systems, or networking). Having an ML/AI background is helpful but not required.

## Tentative Schedule
*This is an evolving list and subject to changes due to the breakneck pace of AI*

| Date       | Topic                                                  | Lecturer       | Slides | Assignment/Summary |
|------------|:------------------------------------------------------:|:--------------:|--------|--------------------|
| Aug 24     | Course Introduction and Logistics                      | Fan Lai        |        |                    |
| Aug 26     | Transformers                                           | Fan Lai    |        |                    |
| Aug 31     | Transformers Deep Dive                                 | Fan Lai        |        |                    |
| Sept 2     | Distributed Training Overview                          | Fan Lai        |        | [DeepSeek-V3 Report (Sec 1-3)](https://arxiv.org/abs/2412.19437) |
| Sept 7     | Data Parallelism                                       | Fan Lai        |        |                    |
| Sept 9     | Tensor Parallelism                                     | Fan Lai        |        | [LlamaRL](https://arxiv.org/abs/2505.24034) |
| Sept 14    | Pipeline Parallelism                                   | Fan Lai        |        |                    |
| Sept 16    | Multi-Dimensional Parallelism                          | Fan Lai        |        | [Alpa](https://www.usenix.org/system/files/osdi22-zheng-lianmin.pdf) |
| Sept 21    | Mixed Precision Training                               | Fan Lai        |        | [Assignment 1 Released](./Homework/hw1/Assignment1.pdf) |
| Sept 23    | **No Class** (Meetings to discuss project ideas)       | Fan Lai        |        |                    |
| Sept 28    | Memory Optimization                                    | Fan Lai        |        | Project Proposal Due |
| Sept 30    | Finetuning Techniques                                  | Fan Lai        |        | [ZeRO-style Data Parallelism](https://arxiv.org/abs/1910.02054) |
| Oct 5      | Course Project Proposal Feedback                       | Fan Lai        |        |                    |
| Oct 7      | Course Project Proposal Feedback                       | Fan Lai        |        |                    |
| Oct 12     | Course Project Proposal Feedback                       | Fan Lai        |        | Assignment 1 Due   |
| Oct 14     | Guest Lecture |  | | |
| Oct 19     | Inference Overview                                     | Fan Lai        |        | [Speculative Decoding](https://arxiv.org/abs/2211.17192) |
| Oct 21     | Batch Serving Techniques                               | Fan Lai        |        | [DistServe](https://arxiv.org/abs/2401.09670) |
| Oct 26     | Paged Attention                                        | Fan Lai        |        | [SGLang](https://arxiv.org/abs/2312.07104) |
| Oct 28     | Adaptive KV                                            | Fan Lai        |        | Assignment 2 Released |
| Nov 2      | Quantization                                           | Fan Lai        |        | [BLASST](https://arxiv.org/abs/2512.12087) |
| Nov 4      | MoE Efficiency                                         | Fan Lai        |        | Mid-semester Report Due |
| Nov 9      | Advanced Topics: RAG Systems                           | Fan Lai        |        | [MoonCake](https://www.usenix.org/conference/fast25/presentation/qin) |
| Nov 11     | Advanced Topics: Agentic AI                         |       |        | [Murakkab](https://www.usenix.org/system/files/osdi26-chaudhry.pdf) |
| Nov 16     | Guest Lecture                                          |   |        | Assignment 2 Due, Assignment 3 Released (for 4-credit students) |
| Nov 18     | Course Project Feedback                                | Fan Lai        |        |                    |
| Nov 21-29  | **Fall Break — No Class**                              |                |        |                    |
| Nov 30     | Course Project Feedback                                |  Fan Lai        |        |                    |
| Dec 2      | Final Presentations                                    |                |        | Assignment 3 Due   |
| Dec 7      | Final Presentations                                    |                |        |                    |
| Dec 9      | Final Presentations                                               |                |        | Final Report Due   |
| Dec 18      | No Class                                               |                |        | Final Report Due   |
 
 ## Tentative Grading
**Groups**:  Panel discussion and research project will be performed in groups of 4-5 students. Form a group and [declare your group's membership and paper preferences](https://forms.gle/A9mrg64SJXVA2FeP9) by **Sept 5**. After this date, we will form groups from the remaining students.

|                         | Weight | 
| ------------------------| :------| 
| [Attendance](#required-reading)           | 10%    | 
| [Panel Discussion](#Post-Lecture-Panel-Discussion)           | 6% (3% + 3%)    | 
| Lab assignments     | 20% (2-3 lab assignments)    | 
| [Reading summary](#paper-summary)           | 24% (opt-in 8 out of 10 readings, 3% each)    | 
| [Final project presentation](#project)          | 15%    | 
| [Project report](#project)   | 25% (5% + 5% + 15%)    |

**Academic integrity**: [The University's Honor Code](https://siebelschool.illinois.edu/academics/honor-code) applies to all activities related to this course. All material you submit in this course (reading summary, project reports, and presentation materials) must be your own. If you use someone else’s material, you must cite them properly. 

**AI Tool Policy**: AI tools may be used for grammar checking and refining initial brainstorms, but the final reviews and codes **must** be authored by the student. Students are responsible for the entire content and must adhere to the Academic Integrity Policy. 

## Policies

### Participation
**Before Each Lecture**: Some lectures may include a required reading. You must submit a summary of the paper by 11:59 PM on the due date.

**During Lectures**: Active participation is crucial for both your own understanding and to improve the overall quality of the course. You are expected to attend **all** lectures (up to 2 absences allowed for legitimate reasons), and more importantly, participate in class discussions. Not everyone must have add something every day, but it is expected that everyone has something to share over the semester.

### Paper Summaries
You need to select and write paper summaries from **8 papers out of 10 listed papers**. The summary should be done independently and include the following contents (five paragraphs):

- P1: The problem the paper is trying to tackle. What's the impact of the work, e.g., why is it an important problem to solve? 
- P2: The main proposed idea(s). 
- P3: A summary of your understanding of different components of the proposed technique, e.g., the purpose of critical design choices.
- P4: Your perceived strengths and weaknesses of the work, e.g., novelty, significance of improvements, quality of the evaluation, easy-to-use.
- P5: Is there room for improvement? If so, what idea do you have for improving the techniques? 

You do not need to write super long paragraphs, as long as you have the key points listed out in each paragraph. You can discuss the paper with other students, but all of your writing work should be your own. DO NOT use AI tools to draft it!

In terms of grading criteria, each summary has 10 points in total. For each review item above, you get:

- 2: The summary item demonstrates a clear understanding of the paper.
- 1: The summary item misses the point of the paper.
- 0: The summary item is missing.

Due to selecting the 8/10 paper summaries, late submissions won't be accepted and will receive 0 points.

### Post-Lecture Panel Discussion 
To foster a deeper understanding of the papers and encourage critical thinking, lectures with paper summary will be followed by a panel discussion. This discussion will involve three distinct roles played by different student groups, simulating an interactive and dynamic scholarly exchange. 

#### Roles and Responsibilities

1. **The Companion (Author) Group**
- Responsibility: As authors, you are expected to defend your paper against critiques, answer questions, and discuss how you might improve or extend your research in the future, akin to writing a rebuttal during the peer-review process.


2. **The Reviewer Group**
- Responsibility: Reviewers critically assess the paper, posing challenging questions and highlighting potential weaknesses or areas for further investigation. 
Your goal is to engage in a constructive critique of the paper, simulating a peer review scenario.

 
3. **Rest of the Class**
- Responsibility: During the panel discussions, feel free to actively **ask questions** and engage in the dialogue. 

The lecturer will also pose challenging questions to both the companion and reviewer groups, so please come well-prepared!

### Project
You will have to complete substantive work an instructor-approved problem and have original contribution. Surveys are not permitted as projects; instead, each project must contain a survey of background and related work.

You must meet the following milestones (unless otherwise specified in future announcements) to ensure a high-quality project at the end of the semester:

* Turn in a 2-page draft proposal ([template](https://www.overleaf.com/read/bsrcbphcvyzc#d075e8)), plus as many pages as needed for references, by **Oct 1**. Remember to include the names and UIUC email addresses of the group members. 
* Each group must schedule project discussion with the instructor during class hours or office hours in the week of **Oct 3** and **Oct 8**.
* Each group must turn in a 3/4-page mid-semester report via email **on or before 6:00PM CST on Nov 7.** 
* Each group must turn in an 8-page final report and your code via email **on or before 6:00PM CST on Dec 18.** The report must be submitted as a PDF file, with formatting similar to that of the papers you've read in the class. The self-contained (i.e., include ALL dependencies) code must be submitted as a zip file. Each zip file containing the code must include a README file with a step-by-step guide on how to compile and run the provided code.
* You can find how to access GPU resources [here](./Resources/cloudlab.md).

### **Acknowledgements**
This course alternates with Prof. Minjia Zhang's [CS 498](https://minjiazhang.github.io/courses/cs-mlsys-498-2025spring.html). Big thanks to Prof. Minjia Zhang!
