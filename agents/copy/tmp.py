def transfer_to_long(self, memory_text):
        # if the new_memory is summarized, otherwise add it into short-term memory
        transfer_flag = False
        transfer_records = [True for _ in range(len(self.short_memories))]
        memory_text, memory_importance, insight_text, insight_importance = [], [], [], []
        for idx, memory in enumerate(self.short_memories):
            # if exceed the enhancement threshold
            if self.enhance_cnt[idx] >= self.enhance_threshold:
                transfer_flag = True
                transfer_records[idx] = True
                # combine all existing related memories to current memory in short-term memories
                enhance_texts = [memory["text"]]
                # do not repeatedly add new_memory memory to summary, so use [:-1].
                for enhance_memory_text in self.enhance_memory_texts[idx][:-1]:
                    enhance_texts.append(enhance_memory_text)
                enhance_texts.append(new_memory_text)
                memory_text.append(memory["text"])
                memory_importance.append(memory["importance"])
                insight_text.append(self.get_short_term_insight(enhance_texts))

        # remove the transferred memories from short-term memories
        if transfer_flag:
            # re-construct the indexes of short-term memories after removing summarized memories
            new_memories = []
            new_enhance_memories = [[] for _ in range(self.capacity)]
            new_enhance_cnt = [0 for _ in range(self.capacity)]
            for idx, memory in enumerate(self.short_memories):
                if not transfer_records[idx]:
                    new_enhance_memories[len(new_memories)] = self.enhance_memory_texts[idx]
                    new_enhance_cnt[len(new_memories)] = self.enhance_cnt[idx]
                    new_memories.append(memory)
            self.short_memories = new_memories
            self.enhance_memory_texts = new_enhance_memories
            self.enhance_cnt = new_enhance_cnt
        return memory_text, memory_importance, insight_text