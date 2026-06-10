using UnityEngine;

public static class TeleopUnityMaterials
{
    private static readonly string[] ShaderNames = {
        "Standard",
        "Universal Render Pipeline/Lit",
        "Unlit/Color",
        "Sprites/Default"
    };

    public static Material Make(Color color)
    {
        Shader shader = null;
        foreach (string name in ShaderNames)
        {
            shader = Shader.Find(name);
            if (shader != null)
            {
                break;
            }
        }

        if (shader == null)
        {
            shader = Shader.Find("Hidden/Internal-Colored");
        }

        var mat = new Material(shader);
        mat.color = color;
        return mat;
    }
}
