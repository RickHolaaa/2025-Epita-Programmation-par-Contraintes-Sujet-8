from datetime import datetime, timedelta
from ortools.sat.python import cp_model
import time
import pandas as pd

# ---------- DATA CLASSES ----------

class Team:
    def __init__(self, name, stadium):
        self.name = name
        self.stadium = stadium

class Stadium:
    def __init__(self, name, unavailable_dates=None):
        self.name = name
        self.unavailable_dates = set()
        if unavailable_dates:
            for date in unavailable_dates:
                if isinstance(date, str):
                    self.unavailable_dates.add(datetime.fromisoformat(date).date())
                elif isinstance(date, datetime):
                    self.unavailable_dates.add(date.date())
                else:
                    try:
                        self.unavailable_dates.add(date)
                    except:
                        pass

    def is_available_on(self, date):
        d = date.date() if isinstance(date, datetime) else date
        return d not in self.unavailable_dates

# ---------- SOLVER ----------

class Schedule:
    def __init__(self, teams, start_date, max_consecutive_away):
        self.teams = teams
        self.start_date = datetime.fromisoformat(start_date)
        self.max_consecutive_away = max_consecutive_away
        self.num_teams = len(teams)
        self.num_days = self.num_teams - 1
        model = cp_model.CpModel()

        valid_days = {
            t: [d for d in range(self.num_days)
                if teams[t].stadium.is_available_on(self.start_date + timedelta(days=d))]
            for t in range(self.num_teams)
        }

        self.X = {}
        for i in range(self.num_teams):
            for j in range(self.num_teams):
                if i == j:
                    continue
                for d in valid_days[i]:
                    self.X[(i, j, d)] = model.NewBoolVar(f"match_{i}_{j}_day{d+1}")

        # Chaque paire se rencontre une fois
        for i in range(self.num_teams):
            for j in range(i + 1, self.num_teams):
                model.Add(
                    sum(self.X.get((i, j, d), 0) for d in range(self.num_days)) +
                    sum(self.X.get((j, i, d), 0) for d in range(self.num_days))
                    == 1
                )

        # Chaque équipe joue un match par jour
        for t in range(self.num_teams):
            for d in range(self.num_days):
                model.Add(
                    sum(self.X.get((t, o, d), 0) for o in range(self.num_teams) if o != t) +
                    sum(self.X.get((o, t, d), 0) for o in range(self.num_teams) if o != t)
                    == 1
                )

        # Max X matchs à l'extérieur consécutifs
        for t in range(self.num_teams):
            for d0 in range(self.num_days - self.max_consecutive_away):
                model.Add(
                    sum(self.X.get((o, t, d), 0)
                        for d in range(d0, d0 + self.max_consecutive_away + 1)
                        for o in range(self.num_teams) if o != t)
                    <= self.max_consecutive_away
                )

        # Variables de statut domicile/extérieur
        is_home = {}
        is_away = {}
        for t in range(self.num_teams):
            for d in range(self.num_days):
                home_expr = sum(self.X.get((t, o, d), 0) for o in range(self.num_teams) if o != t)
                away_expr = sum(self.X.get((o, t, d), 0) for o in range(self.num_teams) if o != t)
                is_home[t, d] = model.NewBoolVar(f"is_home_{t}_day{d}")
                is_away[t, d] = model.NewBoolVar(f"is_away_{t}_day{d}")
                model.Add(home_expr == is_home[t, d])
                model.Add(away_expr == is_away[t, d])

        # Breaks = 2 matchs à domicile ou extérieur de suite
        B = {}
        total_breaks = {}
        for t in range(self.num_teams):
            for d in range(self.num_days - 1):
                b = model.NewBoolVar(f"break_{t}_day{d}")
                home_break = model.NewBoolVar(f"home_break_{t}_{d}")
                away_break = model.NewBoolVar(f"away_break_{t}_{d}")
                model.AddBoolAnd([is_home[t, d], is_home[t, d + 1]]).OnlyEnforceIf(home_break)
                model.AddBoolOr([is_home[t, d].Not(), is_home[t, d + 1].Not()]).OnlyEnforceIf(home_break.Not())
                model.AddBoolAnd([is_away[t, d], is_away[t, d + 1]]).OnlyEnforceIf(away_break)
                model.AddBoolOr([is_away[t, d].Not(), is_away[t, d + 1].Not()]).OnlyEnforceIf(away_break.Not())
                model.AddMaxEquality(b, [home_break, away_break])
                B[t, d] = b
            total_breaks[t] = model.NewIntVar(0, self.num_days, f"total_breaks_{t}")
            model.Add(total_breaks[t] == sum(B[t, d] for d in range(self.num_days - 1)))

        global_breaks = model.NewIntVar(0, self.num_teams * (self.num_days - 1), "global_breaks")
        model.Add(global_breaks == sum(total_breaks[t] for t in range(self.num_teams)))
        model.Minimize(global_breaks)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 600
        result = solver.Solve(model)

        self.total_breaks_value = solver.Value(global_breaks)
        self.schedule = []
        if result in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for d in range(self.num_days):
                matches = []
                for i in range(self.num_teams):
                    for j in range(self.num_teams):
                        if i != j and (i, j, d) in self.X and solver.Value(self.X[(i, j, d)]) == 1:
                            matches.append((i, j))
                self.schedule.append(matches)
        else:
            self.schedule = None

def benchmark_ortools(team_sizes, start_date="2025-04-01", max_consecutive_away=2):
    results = []

    for n in team_sizes:
        print(f"\nBenchmarking for {n} teams...")

        stadiums = [Stadium(f"Stadium {i+1}") for i in range(n)]
        teams = [Team(f"Team {chr(65+i)}", stadiums[i]) for i in range(n)]

        start_time = time.time()
        schedule = Schedule(teams, start_date=start_date, max_consecutive_away=max_consecutive_away)
        end_time = time.time()

        duration = end_time - start_time
        status = "Feasible" if schedule.schedule else "Infeasible"

        results.append({
            "num_teams": n,
            "solve_time_sec": round(duration, 3),
            "status": status,
            "total_breaks": schedule.total_breaks_value if schedule.schedule else None
        })

    return pd.DataFrame(results)

# --- MAIN ---
if __name__ == "__main__":
    team_sizes = [6, 8, 10, 12, 14, 16, 18, 20]
    df = benchmark_ortools(team_sizes)
    print("\n--- OR-Tools Benchmark Results ---")
    print(df.to_markdown(index=False))  # tu peux aussi faire df.to_string() si tu préfères

    with open("result.output", 'w') as f:
        f.write(df.to_markdown(index=False))