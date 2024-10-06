
using StateSpacePathfinding;
using System.Runtime.CompilerServices;

internal class Program
{
    private static void Main(string[] args)
    {
        double[] start = [Math.PI, 0];
        double[] end = [-Math.PI, 0];
        double dt = 0.1;
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
    private const double p = 5.0;
    private const double lookaheadTime = 0.8;
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
        return 3 * Math.Sqrt
            (
                Math.Pow(state[0] + lookaheadTime * state[1] - target[0], 2) +
                Math.Pow(state[1] - target[1], 2)
            );
    }
}