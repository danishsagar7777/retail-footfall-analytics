class FootfallCounter:
    def __init__(self):
        self.entries = 0
        self.exits = 0

    def update(self, event):
        if event == "entry":
            self.entries += 1

        elif event == "exit":
            self.exits += 1

    @property
    def occupancy(self):
        return max(0, self.entries - self.exits)

    def summary(self):
        return {
            "entries": self.entries,
            "exits": self.exits,
            "occupancy": self.occupancy,
        }

