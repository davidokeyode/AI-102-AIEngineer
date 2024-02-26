using System;
using System.IO;
using System.Linq;
using System.Drawing;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;

// Import namespaces
using Microsoft.Azure.CognitiveServices.Vision.Face;
using Microsoft.Azure.CognitiveServices.Vision.Face.Models;


namespace analyze_faces
{
    class Program
    {

        private static FaceClient faceClient;
        static async Task Main(string[] args)
        {
            try
            {
                // Get config settings from AppSettings
                IConfigurationBuilder builder = new ConfigurationBuilder().AddJsonFile("appsettings.json");
                IConfigurationRoot configuration = builder.Build();
                string cogSvcEndpoint = configuration["CognitiveServicesEndpoint"];
                string cogSvcKey = configuration["CognitiveServiceKey"];

                // Authenticate Face client
                ApiKeyServiceClientCredentials credentials = new ApiKeyServiceClientCredentials(cogSvcKey);
                faceClient = new FaceClient(credentials)
                {
                    Endpoint = cogSvcEndpoint
                };

                // Menu for face functions
                Console.WriteLine("1: Detect faces\n2: Compare faces\n3: Train a facial recognition model\n4: Recognize faces\n5: Verify a face\nAny other key to quit");
                Console.WriteLine("Enter a number:");
                string command = Console.ReadLine();
                switch (command)
                {
                    case "1":
                        await DetectFaces("images/people.jpg");
                        break;
                    case "2":
                        string personImage = "images/person1.jpg"; // Also try person2.jpg
                        await CompareFaces(personImage, "images/people.jpg");
                        break;
                    case "3":
                        List<string> names = new List<string>() { "Aisha", "Sama" };
                        await TrainModel("employees_group", "employees", names);
                        break;
                    case "4":
                        await RecognizeFaces("images/people.jpg", "employees_group");
                        break;
                    case "5":
                        await VerifyFace("images/person1.jpg", "Aisha", "employees_group");
                        break;
                    default:
                        break;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }

        static async Task DetectFaces(string imageFile)
        {
            Console.WriteLine($"Detecting faces in {imageFile}");

            // Specify facial features to be retrieved
            IList<FaceAttributeType> features = new FaceAttributeType[]
            {
                FaceAttributeType.Occlusion,
                FaceAttributeType.Blur,
                FaceAttributeType.Glasses
            };

            // Get faces
            using (var imageData = File.OpenRead(imageFile))
            {
                var detected_faces = await faceClient.Face.DetectWithStreamAsync(imageData, returnFaceAttributes: features, returnFaceId: false);

                if (detected_faces.Count() > 0)
                {
                    Console.WriteLine($"{detected_faces.Count()} faces detected.");

                    // Prepare image for drawing
                    Image image = Image.FromFile(imageFile);
                    Graphics graphics = Graphics.FromImage(image);
                    Pen pen = new Pen(Color.LightGreen, 3);
                    Font font = new Font("Arial", 4);
                    SolidBrush brush = new SolidBrush(Color.White);
                    int faceCount = 0;

                    // Draw and annotate each face
                    foreach (var face in detected_faces)
                    {
                        faceCount++;
                        Console.WriteLine($"\nFace number {faceCount}");

                        // Get face properties
                        Console.WriteLine($" - Mouth Occluded: {face.FaceAttributes.Occlusion.MouthOccluded}");
                        Console.WriteLine($" - Eye Occluded: {face.FaceAttributes.Occlusion.EyeOccluded}");
                        Console.WriteLine($" - Blur: {face.FaceAttributes.Blur.BlurLevel}");
                        Console.WriteLine($" - Glasses: {face.FaceAttributes.Glasses}");

                        // Draw and annotate face
                        var r = face.FaceRectangle;
                        Rectangle rect = new Rectangle(r.Left, r.Top, r.Width, r.Height);
                        graphics.DrawRectangle(pen, rect);
                        string annotation = $"Face number {faceCount}";
                        graphics.DrawString(annotation, font, brush, r.Left, r.Top);
                    }

                    // Save annotated image
                    String output_file = "detected_faces.jpg";
                    image.Save(output_file);
                    Console.WriteLine(" Results saved in " + output_file);
                }
            }

        }
        static async Task CompareFaces(string image1, string image2)
        {
            Console.WriteLine($"Comparing faces in {image1} and {image2}");
            // Determine if the face in image 1 is also in image 2
            DetectedFace image_i_face;
            using (var image1Data = File.OpenRead(image1))
            {
                // Get the first face in image 1
                var image1_faces = await faceClient.Face.DetectWithStreamAsync(image1Data);
                if (image1_faces.Count > 0)
                {
                    image_i_face = image1_faces[0];
                    Image img1 = Image.FromFile(image1);
                    Graphics graphics = Graphics.FromImage(img1);
                    Pen pen = new Pen(Color.LightGreen, 3);
                    var r = image_i_face.FaceRectangle;
                    Rectangle rect = new Rectangle(r.Left, r.Top, r.Width, r.Height);
                    graphics.DrawRectangle(pen, rect);
                    String output_file = "face_to_match.jpg";
                    img1.Save(output_file);
                    Console.WriteLine(" Results saved in " + output_file);

                    //Get all the faces in image 2
                    using (var image2Data = File.OpenRead(image2))
                    {
                        var image2Faces = await faceClient.Face.DetectWithStreamAsync(image2Data);

                        // Get faces
                        if (image2Faces.Count > 0)
                        {

                            var image2FaceIds = image2Faces.Select(f => f.FaceId).ToList<Guid?>();
                            var similarFaces = await faceClient.Face.FindSimilarAsync((Guid)image_i_face.FaceId, faceIds: image2FaceIds);
                            var similarFaceIds = similarFaces.Select(f => f.FaceId).ToList<Guid?>();

                            // Prepare image for drawing
                            Image img2 = Image.FromFile(image2);
                            Graphics graphics2 = Graphics.FromImage(img2);
                            Pen pen2 = new Pen(Color.LightGreen, 3);
                            Font font2 = new Font("Arial", 4);
                            SolidBrush brush2 = new SolidBrush(Color.Black);

                            // Draw and annotate each face
                            foreach (var face in image2Faces)
                            {
                                if (similarFaceIds.Contains(face.FaceId))
                                {
                                    // Draw and annotate face
                                    var r2 = face.FaceRectangle;
                                    Rectangle rect2 = new Rectangle(r2.Left, r2.Top, r2.Width, r2.Height);
                                    graphics2.DrawRectangle(pen2, rect2);
                                    string annotation = "Match!";
                                    graphics2.DrawString(annotation, font2, brush2, r2.Left, r2.Top);
                                }
                            }

                            // Save annotated image
                            String output_file2 = "matched_faces.jpg";
                            img2.Save(output_file2);
                            Console.WriteLine(" Results saved in " + output_file2);
                        }
                    }

                }
            }


        }

        static async Task TrainModel(string groupId, string groupName, List<string> imageFolders)
        {
            Console.WriteLine($"Creating model for {groupId}");
            // Delete group if it already exists
            var groups = await faceClient.PersonGroup.ListAsync();
            foreach (var group in groups)
            {
                if (group.PersonGroupId == groupId)
                {
                    await faceClient.PersonGroup.DeleteAsync(groupId);
                }
            }

            // Create the group
            await faceClient.PersonGroup.CreateAsync(groupId, groupName);
            Console.WriteLine("Group created!");

            // Add each person to the group
            Console.Write("Adding people to the group...");
            foreach (var personName in imageFolders)
            {
                // Add the person
                var person = await faceClient.PersonGroupPerson.CreateAsync(groupId, personName);

                // Add multiple photo's of the person
                string[] images = Directory.GetFiles("images/" + personName);
                foreach (var image in images)
                {
                    using (var imageData = File.OpenRead(image))
                    {
                        await faceClient.PersonGroupPerson.AddFaceFromStreamAsync(groupId, person.PersonId, imageData);
                    }
                }

            }

            // Train the model
            Console.WriteLine("Training model...");
            await faceClient.PersonGroup.TrainAsync(groupId);

            // Get the list of people in the group
            Console.WriteLine("Facial recognition model trained with the following people:");
            var people = await faceClient.PersonGroupPerson.ListAsync(groupId);
            foreach (var person in people)
            {
                Console.WriteLine($"-{person.Name}");
            }



        }

        static async Task RecognizeFaces(string imageFile, string groupId)
        {
            Console.WriteLine($"Recognizing faces in {imageFile}");
            // Detect faces in the image
            using (var imageData = File.OpenRead(imageFile))
            {
                var detectedFaces = await faceClient.Face.DetectWithStreamAsync(imageData);

                // Get faces
                if (detectedFaces.Count > 0)
                {

                    // Get a list of face IDs
                    var faceIds = detectedFaces.Select(f => f.FaceId).ToList<Guid?>();

                    // Identify the faces in the people group
                    var recognizedFaces = await faceClient.Face.IdentifyAsync(faceIds.Cast<Guid>().ToList(), groupId);

                    // Get names for recognized faces
                    var faceNames = new Dictionary<Guid?, string>();
                    if (recognizedFaces.Count > 0)
                    {
                        foreach (var face in recognizedFaces)
                        {
                            var person = await faceClient.PersonGroupPerson.GetAsync(groupId, face.Candidates[0].PersonId);
                            Console.WriteLine($"-{person.Name}");
                            faceNames.Add(face.FaceId, person.Name);
                        }
                    }


                    // Annotate faces in image
                    Image image = Image.FromFile(imageFile);
                    Graphics graphics = Graphics.FromImage(image);
                    Pen penYes = new Pen(Color.LightGreen, 3);
                    Pen penNo = new Pen(Color.Magenta, 3);
                    Font font = new Font("Arial", 4);
                    SolidBrush brush = new SolidBrush(Color.Cyan);
                    foreach (var face in detectedFaces)
                    {
                        var r = face.FaceRectangle;
                        Rectangle rect = new Rectangle(r.Left, r.Top, r.Width, r.Height);
                        if (faceNames.ContainsKey(face.FaceId))
                        {
                            // If the face is recognized, annotate in green with the name
                            graphics.DrawRectangle(penYes, rect);
                            string personName = faceNames[face.FaceId];
                            graphics.DrawString(personName, font, brush, r.Left, r.Top);
                        }
                        else
                        {
                            // Otherwise, just annotate the unrecognized face in magenta
                            graphics.DrawRectangle(penNo, rect);
                        }
                    }

                    // Save annotated image
                    String output_file = "recognized_faces.jpg";
                    image.Save(output_file);
                    Console.WriteLine("Results saved in " + output_file);
                }
            }

        }

        static async Task VerifyFace(string personImage, string personName, string groupId)
        {
            Console.WriteLine($"Verifying the person in {personImage} is {personName}");

            string result = "Not verified";

            // Get the ID of the person from the people group
            var people = await faceClient.PersonGroupPerson.ListAsync(groupId);
            foreach (var person in people)
            {
                if (person.Name == personName)
                {
                    Guid personId = person.PersonId;

                    // Get the first face in the image
                    using (var imageData = File.OpenRead(personImage))
                    {
                        var faces = await faceClient.Face.DetectWithStreamAsync(imageData);
                        if (faces.Count > 0)
                        {
                            Guid faceId = (Guid)faces[0].FaceId;

                            //We have a face and an ID. Do they match?
                            var verification = await faceClient.Face.VerifyFaceToPersonAsync(faceId, personId, groupId);
                            if (verification.IsIdentical)
                            {
                                result = "Verified";
                            }
                        }
                    }
                }
            }

            // print the result
            Console.WriteLine(result);
        }
    }
}
