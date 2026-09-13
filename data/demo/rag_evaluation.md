# RAG evaluation

A retrieval-augmented generation system should evaluate retrieval and generation separately. Retrieval metrics reveal whether the necessary evidence reached the model. Answer metrics reveal whether the model used that evidence correctly. Without this separation, a correct answer may hide failed retrieval and an incorrect answer may be blamed on the wrong component.

