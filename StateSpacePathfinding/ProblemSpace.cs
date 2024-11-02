using System.Collections;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Net;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;

namespace StateSpacePathfinding;


public class ProblemSpace
{
    public readonly int stateSpaceDims;
    public readonly int controlSpaceDims;
    private readonly StateDerivative stateDerivative;
    private readonly CostFunction costFunction;
    private readonly Heuristic heuristic;
    private readonly Vector[] controlSearchPattern;
    private const double gridSize = 0.075;

    public ProblemSpace(
        int stateSpaceDims, int controlSpaceDims, StateDerivative stateDerivative, 
        CostFunction costFunction, Heuristic heuristic, double[][] controlSearchPattern)
    {
        this.stateSpaceDims = stateSpaceDims;
        this.controlSpaceDims = controlSpaceDims;
        this.stateDerivative = stateDerivative;
        this.costFunction = costFunction;
        this.heuristic = heuristic;

        if (controlSearchPattern.Length < 1)
        {
            throw new ArgumentException("Control space search pattern must have at least one vector.");
        }

        foreach (double[] v in controlSearchPattern)
        {
            if (v.Length != controlSpaceDims) { throw new ArgumentException("Control space search pattern must have vectors with the correct number of dimensions."); }
        }


        this.controlSearchPattern = new Vector[controlSearchPattern.Length];

        for(int i = 0; i < controlSearchPattern.Length; i++)
        {
            this.controlSearchPattern[i] = new(controlSearchPattern[i]);
        }
    }

    public delegate double[] StateDerivative(double[] state, double[] control, double dt);
    public delegate double CostFunction(double[] state, double[] control, double dt);
    public delegate double Heuristic(double[] state, double[] target);

    public double[][] PathBetween(double[] start, double[] end, double dt)
    {
        return PathBetween(start, end, dt, out _);
    }

    public double[][] PathBetween(double[] start, double[] end, double dt, out double[][] states)
    {
        if (start.Length != stateSpaceDims || end.Length != stateSpaceDims)
        {
            throw new ArgumentException("Start and end points must be in state space.");
        }

        PriorityQueue<Edge, double> priorityQueue = new();
        HashSet<GridCoords> grid = new();

        Node root = new(new Vector(start), null, this, 0);
        root.GenerateEdges(dt, grid);
        foreach (Edge edge in root.edges)
        {
            priorityQueue.Enqueue(edge, edge.cost + heuristic(edge.end.state, end));
        }

        int counter = 0;

        while ((priorityQueue.Peek().end.state - new Vector(end)).Magnitude > 0.075 && counter < 40000)
        {
            counter++;

            Edge currentBest = priorityQueue.Dequeue();

            currentBest.end.GenerateEdges(dt, grid);
            foreach (Edge edge in currentBest.end.edges)
            {
                priorityQueue.Enqueue(edge, edge.cost + heuristic(edge.end.state, end));
            }
        }

        LinkedList<Vector> controls = new();
        LinkedList<Vector> statesList = new();

        for (Edge current = priorityQueue.Peek(); current.start.prev != null; current = current.start.prev)
        {
            controls.AddFirst(current.control);
            statesList.AddFirst(current.end.state);
        }

        Console.Write("Searched Nodes: [");
        root.PrintAll();
        Console.WriteLine("]");

        double[][] resultControls = new double[controls.Count][];
        double[][] resultStates = new double[controls.Count][];
        IEnumerator<Vector> enumeratorControls = controls.GetEnumerator();
        IEnumerator<Vector> enumeratorStates = statesList.GetEnumerator();
        for (int i = 0; i < controls.Count; i++)
        {
            enumeratorControls.MoveNext();
            resultControls[i] = enumeratorControls.Current;

            enumeratorStates.MoveNext();
            resultStates[i] = enumeratorStates.Current;
        }

        states = resultStates;
        return resultControls;
    }

    [DebuggerDisplay("{ToString()}")]
    private readonly struct Vector
    {
        public readonly double[] values;
        public double Magnitude
        {
            get
            {
                double total = 0;
                foreach (double value in values)
                {
                    total += value * value;
                }
                return Math.Sqrt(total);
            }
        }

        public Vector(double[] values)
        {
            this.values = values;
        }

        public static implicit operator double[](Vector v)
        {
            return v.values;
        }

        public static Vector operator +(Vector left, Vector right)
        {
            double[] result = new double[left.values.Length];
            
            for (int i = 0;  i < left.values.Length; i++)
            {
                result[i] = left.values[i] + right.values[i];
            }

            return new(result);
        }

        public static Vector operator -(Vector left, Vector right)
        {
            double[] result = new double[left.values.Length];

            for (int i = 0; i < left.values.Length; i++)
            {
                result[i] = left.values[i] - right.values[i];
            }

            return new(result);
        }

        public override string ToString()
        {
            StringBuilder result = new("[");
            foreach (double value in values)
            {
                result.Append(value.ToString() + ", ");
            }
            result.Append("]");

            return result.ToString();
        }
    }

    [DebuggerDisplay("state = {state}; Edge count = {edges.Count}; cost = {cost}")]
    private class Node
    {
        public readonly Vector state;
        private readonly ProblemSpace enclosing;
        public readonly List<Edge> edges;
        public readonly Edge? prev;
        public readonly double cost;

        public Node(Vector state, Edge? prev, ProblemSpace enclosing, double cost)
        {
            this.state = state;
            this.enclosing = enclosing;
            this.prev = prev;
            edges = new();
            this.cost = cost;
        }

        public void GenerateEdges(double dt, HashSet<GridCoords> grid)
        {
            foreach (Vector u in enclosing.controlSearchPattern)
            {
                Edge newEdge = new Edge(this, u, enclosing, dt);
                if (!grid.Contains(new(newEdge.end.state)))
                {
                    edges.Add(newEdge);
                    grid.Add(new(newEdge.end.state));
                }
            }
        }

        public void PrintAll()
        {
            if (new Random().Next(0,20) == 0)
            Console.Write("({0}),", string.Join(", ", (double[]) state));
            foreach (Edge e in edges)
            {
                e.end.PrintAll();
            }
        }
    }

    [DebuggerDisplay("Start = {start.state}; end = {end?.state}; cost = {cost}")]
    private class Edge
    {
        public readonly Node start;
        public readonly Node end;
        public readonly Vector control;
        private readonly double dt;

        private readonly ProblemSpace enclosing;

        public readonly double cost;

        public Edge(Node start, Vector control, ProblemSpace enclosing, double dt)
        {
            this.start = start;
            this.control = control;
            this.enclosing = enclosing;
            this.dt = dt;

            cost = start.cost + enclosing.costFunction(start.state, control, dt);
            end = new Node(start.state + new Vector(enclosing.stateDerivative(start.state, control, dt)), this, enclosing, cost);
        }
    }

    private struct GridCoords(int[] coords)
    {
        private readonly int[] coords = coords;

        public GridCoords(double[] decimalCoords): this(new int[decimalCoords.Length])
        {
            for (int i = 0; i < decimalCoords.Length; i++)
            {
                coords[i] = (int)(decimalCoords[i] / gridSize);
            }
        }

        public override int GetHashCode() =>
            coords.Aggregate(0, (total, next) => HashCode.Combine(total, next));

        public override bool Equals([NotNullWhen(false)] object? obj)
        {
            if (obj is GridCoords other) {
                if (coords.Length != other.coords.Length) return false;

                for (int i = 0; i < coords.Length; i++)
                {
                    if (coords[i] != other.coords[i]) return false;
                }

                return true;
            } else
            {
                return false;
            }
        }

        public static bool operator ==(GridCoords lhs, GridCoords rhs) { return lhs.Equals(rhs); }
        public static bool operator !=(GridCoords lhs, GridCoords rhs) { return !lhs.Equals(rhs); }
    }
}