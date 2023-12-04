using Azure;
using System;
using Azure.AI.TextAnalytics;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.IO;
using Microsoft.Extensions.Configuration;

namespace Example
{
    class Program
    {
        
        static async Task Main(string[] args)
        {
            try
            {
                // Get config settings from appsettings.json
                IConfigurationBuilder builder = new ConfigurationBuilder().AddJsonFile("appsettings.json");
                IConfigurationRoot configuration = builder.Build();
                string cogSvcEndpoint = configuration["CognitiveServicesEndpoint"];
                string cogSvcKey = configuration["CognitiveServiceKey"];

                // Create client using endpoint and key
                AzureKeyCredential credentials = new AzureKeyCredential(cogSvcKey);
                Uri endpoint = new Uri(cogSvcEndpoint);
                TextAnalyticsClient client = new TextAnalyticsClient(endpoint, credentials);

                // Read text files from the healthcare_texts folder
                var folderPath = Path.GetFullPath("./healthcare_texts");
                DirectoryInfo folder = new DirectoryInfo(folderPath);
                foreach (var file in folder.GetFiles("*.txt"))
                {
                    // Read the file contents
                    Console.WriteLine("\n-------------\n" + file.Name);
                    StreamReader sr = file.OpenText();
                    var text = sr.ReadToEnd();
                    sr.Close();

                    // Analyze healthcare-related text
                    await HealthExample(client, text);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }
        
        // Example method for extracting information from healthcare-related text 
        static async Task HealthExample(TextAnalyticsClient client, string text)
        {
            List<string> batchInput = new List<string>() { text };
            AnalyzeHealthcareEntitiesOperation healthOperation = await client.StartAnalyzeHealthcareEntitiesAsync(batchInput);
            await healthOperation.WaitForCompletionAsync();

            await foreach (AnalyzeHealthcareEntitiesResultCollection documentsInPage in healthOperation.Value)
            {
                Console.WriteLine($"Results of Azure Text Analytics for health async model, version: \"{documentsInPage.ModelVersion}\"");
                Console.WriteLine("");

                foreach (AnalyzeHealthcareEntitiesResult entitiesInDoc in documentsInPage)
                {
                    if (!entitiesInDoc.HasError)
                    {
                        foreach (var entity in entitiesInDoc.Entities)
                        {
                            Console.WriteLine($"  Entity: {entity.Text}");
                            Console.WriteLine($"  Category: {entity.Category}");
                            Console.WriteLine($"  Offset: {entity.Offset}");
                            Console.WriteLine($"  Length: {entity.Length}");
                            Console.WriteLine($"  NormalizedText: {entity.NormalizedText}");
                        }
                        Console.WriteLine($"  Found {entitiesInDoc.EntityRelations.Count} relations in the current document:");
                        Console.WriteLine("");

                        foreach (HealthcareEntityRelation relations in entitiesInDoc.EntityRelations)
                        {
                            Console.WriteLine($"    Relation: {relations.RelationType}");
                            Console.WriteLine($"    For this relation there are {relations.Roles.Count} roles");

                            foreach (HealthcareEntityRelationRole role in relations.Roles)
                            {
                                Console.WriteLine($"      Role Name: {role.Name}");
                                Console.WriteLine($"      Associated Entity Text: {role.Entity.Text}");
                                Console.WriteLine($"      Associated Entity Category: {role.Entity.Category}");
                                Console.WriteLine("");
                            }
                            Console.WriteLine("");
                        }
                    }
                    else
                    {
                        Console.WriteLine("  Error!");
                        Console.WriteLine($"  Document error code: {entitiesInDoc.Error.ErrorCode}.");
                        Console.WriteLine($"  Message: {entitiesInDoc.Error.Message}");
                    }
                    Console.WriteLine("");
                }
            }
        }

    }
}
