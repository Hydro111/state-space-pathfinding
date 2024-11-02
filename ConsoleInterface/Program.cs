
using StateSpacePathfinding;
using System.Runtime.CompilerServices;

internal class Program
{
    private static void Main(string[] args)
    {
        double[] start = [2*Math.PI, 0];
        double[] end = [0, 0];
        double dt = 0.075;
        ProblemSpace problemSpace = new(2, 1, StateDerivative, CostFunction, Heuristic, searchPattern);

        double[][] result = problemSpace.PathBetween(start, end, dt, out double[][] statePath);

        Console.WriteLine("Controls: ");
        Console.WriteLine(result.ToString());

        Console.WriteLine("Simulated Path: ");
        double[] currentState = start;
        foreach (double[] control in result)
        {
            Console.Write("({0}),", string.Join(", ", currentState));

            double[] stateDif = StateDerivative(currentState, control, dt);

            currentState[0] += stateDif[0];
            currentState[1] += stateDif[1];
        }
        Console.WriteLine(currentState.ToString());

        Console.WriteLine("Predicted Path: ");
        foreach (double[] state in statePath)
        {
            Console.Write("({0}),", string.Join(", ", state));
        }
    }

    private const double g = 9.8;
    private const double l = 1;
    private const double m = 9.8;
    private const double p = 25.0;
    private const double lookaheadTime = 0.4;
    private static readonly double[][] searchPattern =
    {
        [ 1.0],
        [ 0.0],
        [-1.0]
    };

    private static double[] StateDerivative(double[] state, double[] control, double dt)
    {
        return 
        [
            state[1] * dt, 

            (
                (p * control[0] / m / l) - 
                (g * Math.Sin(state[0]) / l)
            ) * dt
        ];
    }

    private static double CostFunction(double[] state, double[] control, double dt)
    {
        return dt + dt * Math.Abs(control[0] / 10000);
    }

    private static double Heuristic(double[] state, double[] target)
    {
        double[] futureState = Add(state, StateDerivative(state, new double[1], lookaheadTime));
        return 2 * Math.Sqrt
            (
                1 * Math.Pow(futureState[0] - target[0], 2) +
                1 * Math.Pow(futureState[1] - target[1], 2)
            );
    }

    private static double[] Add(double[] one, double[] two)
    {
        double[] result = new double[one.Length];

        for (int i = 0; i < one.Length; i++)
        {
            result[i] = one[i] + two[i];
        }

        return result;
    }
}